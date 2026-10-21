//! Automatic choice of how many threads score genomes (`quat-dag-evolve --auto-workers`).
//!
//! Hill climbing on measured throughput. Time is cut into windows of completed genomes, each
//! window run at one thread count. A cycle runs three counts (the current one, one step up, one
//! step down) in rotation, three windows each, so slow drifts in genome cost hit all of them
//! alike. At the end of a cycle the count moves to the SMALLEST candidate whose throughput is
//! within 4% of the best: when more threads no longer help (the GPU is the bottleneck) it backs
//! off instead of hogging cores, and it climbs only for a real gain. Pure logic with injected
//! time, so it is testable without a GPU.

use std::collections::BTreeMap;

/// A count within this fraction of the best throughput counts as equally good.
const TIE: f64 = 0.96;
/// Windows per candidate in one cycle.
const REPS: usize = 3;

#[derive(Debug, Clone)]
pub struct AutoTune {
    max: usize,
    cur: usize,
    step: usize,
    seq: Vec<usize>,
    pos: usize,
    acc: BTreeMap<usize, (f64, f64)>, // threads -> (items, seconds)
    win_n: usize,
    win_items: usize,
    win_start: Option<f64>,
    /// Set when a cycle moved the count: (from, to, throughput at `from`, throughput at `to`), items/s.
    pub last_move: Option<(usize, usize, f64, f64)>,
}

impl AutoTune {
    pub fn new(start: usize, max: usize) -> AutoTune {
        let max = max.max(1);
        let mut t = AutoTune { max, cur: start.clamp(1, max), step: 2, seq: Vec::new(), pos: 0, acc: BTreeMap::new(), win_n: 1, win_items: 0, win_start: None, last_move: None };
        t.restart_cycle();
        t
    }

    /// Thread count to run with right now.
    pub fn current(&self) -> usize { self.win_n }
    /// The count the tuner considers best so far (what it rests on between probes).
    pub fn base(&self) -> usize { self.cur }

    /// Someone set a count by hand: continue the search from there.
    pub fn set_base(&mut self, n: usize) {
        self.cur = n.clamp(1, self.max);
        self.step = 2;
        self.acc.clear();
        self.restart_cycle();
    }

    fn restart_cycle(&mut self) {
        let up = (self.cur + self.step).min(self.max);
        let down = self.cur.saturating_sub(self.step).max(1);
        self.seq.clear();
        for _ in 0..REPS {
            self.seq.push(self.cur);
            if up != self.cur { self.seq.push(up); }
            if down != self.cur { self.seq.push(down); }
        }
        self.pos = 0;
        self.win_n = self.seq[0];
        self.win_items = 0;
        self.win_start = None;
    }

    fn window_target(&self) -> usize { (2 * self.win_n).max(12) }

    /// A batch of genomes starts: its first window starts now.
    pub fn batch_start(&mut self, now: f64) {
        self.win_start = Some(now);
        self.win_items = 0;
    }

    /// The batch is over: a half-finished window would be unfairly short, drop it.
    pub fn batch_end(&mut self) {
        self.win_start = None;
        self.win_items = 0;
    }

    /// One genome finished at time `now` (seconds). Returns true when the window ended and
    /// `current()` may have changed.
    pub fn item_done(&mut self, now: f64) -> bool {
        let Some(start) = self.win_start else { return false };
        self.win_items += 1;
        if self.win_items < self.window_target() { return false; }
        let secs = (now - start).max(1e-9);
        let e = self.acc.entry(self.win_n).or_insert((0.0, 0.0));
        e.0 += self.win_items as f64;
        e.1 += secs;
        self.pos += 1;
        if self.pos >= self.seq.len() { self.decide(); }
        self.win_n = self.seq[self.pos];
        self.win_items = 0;
        self.win_start = Some(now);
        true
    }

    fn decide(&mut self) {
        let thr: Vec<(usize, f64)> = self.acc.iter().filter(|(_, (_, s))| *s > 0.0).map(|(n, (i, s))| (*n, i / s)).collect();
        let best = thr.iter().map(|t| t.1).fold(0.0, f64::max);
        let chosen = thr.iter().find(|t| t.1 >= TIE * best).map(|t| t.0).unwrap_or(self.cur); // ascending n: smallest first
        if chosen != self.cur {
            let at = |n: usize| thr.iter().find(|t| t.0 == n).map(|t| t.1).unwrap_or(0.0);
            self.last_move = Some((self.cur, chosen, at(self.cur), at(chosen)));
            self.cur = chosen;
            self.step = 2;
        } else {
            self.step = 1;
        }
        self.acc.clear();
        self.restart_cycle();
        // restart_cycle reset pos to 0; the caller reads seq[pos] next.
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed `batches` batches of 130 genomes through the tuner on a machine whose throughput is
    /// `thr(n)` genomes per second at n threads; returns the count it rests on.
    fn run(start: usize, max: usize, thr: impl Fn(usize) -> f64, batches: usize) -> AutoTune {
        let mut t = AutoTune::new(start, max);
        let mut now = 0.0;
        for _ in 0..batches {
            t.batch_start(now);
            for _ in 0..130 {
                now += 1.0 / thr(t.current());
                t.item_done(now);
            }
            t.batch_end();
            now += 5.0; // breeding between batches
        }
        t
    }

    #[test]
    fn backs_off_when_more_threads_stop_helping() {
        // GPU-bound: throughput saturates at 6 threads.
        let t = run(12, 16, |n| n.min(6) as f64, 40);
        assert!((6..=7).contains(&t.base()), "should settle at the knee, got {}", t.base());
    }

    #[test]
    fn climbs_while_more_threads_help() {
        let t = run(2, 16, |n| n as f64, 60);
        assert!(t.base() >= 14, "linear scaling should climb to the top, got {}", t.base());
    }

    #[test]
    fn stays_put_on_a_flat_curve_at_the_smallest_equal_count() {
        let t = run(8, 16, |_| 10.0, 40);
        assert!(t.base() <= 3, "equal throughput everywhere → fewest threads, got {}", t.base());
    }

    #[test]
    fn manual_set_restarts_the_search_from_there() {
        let mut t = run(12, 16, |n| n.min(6) as f64, 20);
        t.set_base(15);
        assert_eq!(t.base(), 15);
        assert_eq!(t.current(), 15);
    }

    #[test]
    fn survives_noisy_item_costs() {
        // Genome cost varies ±40% item to item; the knee is still at 6.
        let mut t = AutoTune::new(12, 16);
        let mut now = 0.0;
        let mut seed = 12345u64;
        let mut hist = Vec::new();
        for _ in 0..80 {
            t.batch_start(now);
            for _ in 0..130 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let noise = 1.0 + 0.4 * (((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0);
                now += noise / (t.current().min(6) as f64);
                t.item_done(now);
            }
            t.batch_end();
            hist.push(t.base());
        }
        let late = &hist[40..];
        assert!(late.iter().all(|&n| (5..=9).contains(&n)), "should hover near the knee, got {late:?}");
    }
}
