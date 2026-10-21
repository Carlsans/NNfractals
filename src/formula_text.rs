//! Editable text form of an expression-DAG program: a lossless printer and
//! the parser that reads it back.
//!
//! `Genome::formula_expr` is a display string — constants rounded to two
//! decimals, glyphs like `|x|ʙs` — and nothing can parse it. The anim
//! viewer's live formula editor needs a text the user can change and have
//! turned back into a program, so this module defines one syntax used in
//! both directions:
//!
//! ```text
//! z, c            leaves (current iterate, parameter)
//! 0.25  0.5i  i   real / imaginary literals; (0.2+0.1i) is one constant
//! + - * · /       binary ops (* and · are the same)
//! -x              unary minus
//! x^2 x^3 x^4     SQR / CUBE / QUART (also x² x³ x⁴); x^-1 is RECIP
//! sin cos exp log tanh conj recip absfold absre absim normz sqr cube quart, bulb(a, p),
//! kbc(x, y) = the constant (0, 0, x, y)
//! ```
//!
//! Round-trip guarantee: `parse_program(&program_to_text(p))` evaluates
//! identically to `p` (the quaternion product is non-commutative, so operand
//! ORDER is always preserved). Node layout may differ: unreachable nodes
//! (introns) are dropped and identical subexpressions are merged.

use crate::formula::{op, OpNode, N_SLOTS};

#[derive(Clone, Debug, PartialEq)]
pub struct ParseError {
    /// Character (not byte) offset into the input where the error was found.
    pub pos: usize,
    pub msg: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "at column {}: {}", self.pos + 1, self.msg)
    }
}

// ── Printer ─────────────────────────────────────────────────────────────────

// Precedence levels, loosest first. A child printed into a slot that demands
// a tighter level than it has gets parenthesised.
const LV_ADD: u8 = 1;
const LV_MUL: u8 = 2;
const LV_UNARY: u8 = 3;
const LV_POSTFIX: u8 = 4;
const LV_ATOM: u8 = 5;

fn fmt_num(v: f32) -> String {
    // Rust's `{}` for f32 is the shortest string that parses back to the
    // same f32, and never uses exponent notation.
    if v.is_finite() { format!("{v}") } else { "0".to_string() }
}

/// A constant as text plus its precedence level.
fn const_text(re: f32, im: f32) -> (String, u8) {
    let re = if re.is_finite() { re } else { 0.0 };
    let im = if im.is_finite() { im } else { 0.0 };
    if im == 0.0 {
        if re >= 0.0 && !(re == 0.0 && re.is_sign_negative()) {
            (fmt_num(re), LV_ATOM)
        } else {
            (format!("(-{})", fmt_num(-re)), LV_ATOM)
        }
    } else if re == 0.0 {
        if im > 0.0 { (format!("{}i", fmt_num(im)), LV_ATOM) } else { (format!("(-{}i)", fmt_num(-im)), LV_ATOM) }
    } else {
        let sign = if im < 0.0 { '-' } else { '+' };
        let r = if re < 0.0 { format!("-{}", fmt_num(-re)) } else { fmt_num(re) };
        (format!("({r}{sign}{}i)", fmt_num(im.abs())), LV_ATOM)
    }
}

fn wrap((s, lv): (String, u8), need: u8) -> String {
    if lv < need { format!("({s})") } else { s }
}

/// Lossless, parseable text of `prog`, printed from its root.
///
/// Mirrors `quat_dag::eval_program_quat` exactly where it is defensive:
/// only the first `N_SLOTS` nodes run, the root is the last of those, an
/// operand that does not point at an earlier node reads zero, and an
/// unknown opcode yields zero.
pub fn program_to_text(prog: &[OpNode]) -> String {
    let n = prog.len().min(N_SLOTS);
    if n == 0 {
        return "0".to_string();
    }
    let mut memo: Vec<Option<(String, u8)>> = vec![None; n];
    node_text(prog, n - 1, &mut memo).0
}

fn node_text(prog: &[OpNode], i: usize, memo: &mut Vec<Option<(String, u8)>>) -> (String, u8) {
    if let Some(t) = &memo[i] {
        return t.clone();
    }
    let node = prog[i];
    let operand = |idx: u8, memo: &mut Vec<Option<(String, u8)>>| -> (String, u8) {
        let idx = (idx as usize).min(N_SLOTS - 1);
        if idx < i { node_text(prog, idx, memo) } else { ("0".to_string(), LV_ATOM) }
    };
    let func = |name: &str, a: (String, u8)| (format!("{name}({})", a.0), LV_ATOM);
    let out = match node.op {
        op::Z => ("z".to_string(), LV_ATOM),
        op::C => ("c".to_string(), LV_ATOM),
        op::CONST => const_text(node.kre, node.kim),
        op::CONSTBC => {
            let f = |v: f32| { let v = if v.is_finite() { v } else { 0.0 }; if v < 0.0 { format!("-{}", fmt_num(-v)) } else { fmt_num(v) } };
            (format!("kbc({}, {})", f(node.kre), f(node.kim)), LV_ATOM)
        }
        op::SQR | op::CUBE | op::QUART => {
            let a = operand(node.a, memo);
            let p = match node.op { op::SQR => 2, op::CUBE => 3, _ => 4 };
            (format!("{}^{p}", wrap(a, LV_ATOM)), LV_POSTFIX)
        }
        op::RECIP => func("recip", operand(node.a, memo)),
        op::SIN => func("sin", operand(node.a, memo)),
        op::COS => func("cos", operand(node.a, memo)),
        op::EXP => func("exp", operand(node.a, memo)),
        op::LOG => func("log", operand(node.a, memo)),
        op::TANH => func("tanh", operand(node.a, memo)),
        op::CONJ => func("conj", operand(node.a, memo)),
        op::ABSFOLD => func("absfold", operand(node.a, memo)),
        op::ABSRE => func("absre", operand(node.a, memo)),
        op::ABSIM => func("absim", operand(node.a, memo)),
        op::NORMZ => func("normz", operand(node.a, memo)),
        op::BULB => (format!("bulb({}, {})", operand(node.a, memo).0, operand(node.b, memo).0), LV_ATOM),
        op::ADD | op::SUB => {
            let a = operand(node.a, memo);
            let b = operand(node.b, memo);
            let sym = if node.op == op::ADD { '+' } else { '-' };
            (format!("{} {sym} {}", wrap(a, LV_ADD), wrap(b, LV_MUL)), LV_ADD)
        }
        op::MUL | op::DIV => {
            let a = operand(node.a, memo);
            let b = operand(node.b, memo);
            let sym = if node.op == op::MUL { '*' } else { '/' };
            (format!("{}{sym}{}", wrap(a, LV_MUL), wrap(b, LV_UNARY)), LV_MUL)
        }
        _ => ("0".to_string(), LV_ATOM),
    };
    memo[i] = Some(out.clone());
    out
}

// ── Parser ──────────────────────────────────────────────────────────────────

/// Builds nodes with hash-consing: pushing a node identical to an existing
/// one returns the existing index, so repeated subexpressions are shared.
struct Builder {
    nodes: Vec<OpNode>,
}

impl Builder {
    fn push(&mut self, mut n: OpNode) -> usize {
        // Normalise fields the op does not read, so equal nodes hash equal.
        if op::arity(n.op) < 2 { n.b = 0; }
        if op::arity(n.op) < 1 { n.a = 0; }
        if n.op != op::CONST && n.op != op::CONSTBC { n.kre = 0.0; n.kim = 0.0; }
        let same = |m: &OpNode| {
            m.op == n.op && m.a == n.a && m.b == n.b
                && m.kre.to_bits() == n.kre.to_bits() && m.kim.to_bits() == n.kim.to_bits()
        };
        if let Some(i) = self.nodes.iter().position(same) {
            return i;
        }
        self.nodes.push(n);
        self.nodes.len() - 1
    }

    fn konst(&mut self, re: f32, im: f32) -> usize {
        self.push(OpNode { op: op::CONST, a: 0, b: 0, kre: re, kim: im })
    }

    fn as_const(&self, i: usize) -> Option<(f32, f32)> {
        let n = self.nodes[i];
        (n.op == op::CONST).then_some((n.kre, n.kim))
    }

    fn unary(&mut self, o: u8, a: usize) -> usize {
        self.push(OpNode { op: o, a: a as u8, b: 0, kre: 0.0, kim: 0.0 })
    }

    fn binary(&mut self, o: u8, a: usize, b: usize) -> usize {
        // Fold `real ± imag` into one constant. Only this exact shape is
        // folded: with one side purely real and the other purely imaginary
        // the result is exact, so the printed form `(0.2+0.1i)` reads back
        // bit-identically. Folding general constant arithmetic would round
        // in f32 where the evaluator computes in f64.
        if o == op::ADD || o == op::SUB {
            if let (Some((ar, ai)), Some((br, bi))) = (self.as_const(a), self.as_const(b)) {
                let s = if o == op::ADD { 1.0 } else { -1.0 };
                if ai == 0.0 && br == 0.0 {
                    return self.konst(ar, s * bi);
                }
            }
        }
        self.push(OpNode { op: o, a: a as u8, b: b as u8, kre: 0.0, kim: 0.0 })
    }

    /// Unary minus: folded into a constant, otherwise `0 - x`.
    fn neg(&mut self, a: usize) -> usize {
        if let Some((r, i)) = self.as_const(a) {
            return self.konst(-r, -i);
        }
        let zero = self.konst(0.0, 0.0);
        self.push(OpNode { op: op::SUB, a: zero as u8, b: a as u8, kre: 0.0, kim: 0.0 })
    }
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
    b: Builder,
}

fn func_op(name: &str) -> Option<u8> {
    Some(match name {
        "sin" => op::SIN, "cos" => op::COS, "exp" => op::EXP, "log" => op::LOG,
        "tanh" => op::TANH, "conj" => op::CONJ, "recip" => op::RECIP,
        "absfold" => op::ABSFOLD, "absre" => op::ABSRE, "absim" => op::ABSIM,
        "normz" => op::NORMZ, "bulb" => op::BULB, "sqr" => op::SQR, "cube" => op::CUBE, "quart" => op::QUART,
        _ => return None,
    })
}

impl Parser {
    fn err<T>(&self, pos: usize, msg: impl Into<String>) -> Result<T, ParseError> {
        Err(ParseError { pos, msg: msg.into() })
    }

    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_ws();
        self.chars.get(self.pos).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) { self.pos += 1; true } else { false }
    }

    fn expr(&mut self) -> Result<usize, ParseError> {
        let mut lhs = self.term()?;
        loop {
            let o = match self.peek() {
                Some('+') => op::ADD,
                Some('-') => op::SUB,
                _ => return Ok(lhs),
            };
            self.pos += 1;
            let rhs = self.term()?;
            lhs = self.b.binary(o, lhs, rhs);
        }
    }

    fn term(&mut self) -> Result<usize, ParseError> {
        let mut lhs = self.unary()?;
        loop {
            let o = match self.peek() {
                Some('*') | Some('·') => op::MUL,
                Some('/') => op::DIV,
                _ => return Ok(lhs),
            };
            self.pos += 1;
            let rhs = self.unary()?;
            lhs = self.b.binary(o, lhs, rhs);
        }
    }

    fn unary(&mut self) -> Result<usize, ParseError> {
        if self.eat('-') {
            let a = self.unary()?;
            return Ok(self.b.neg(a));
        }
        if self.eat('+') {
            return self.unary();
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<usize, ParseError> {
        let a = self.atom()?;
        let o = match self.peek() {
            Some('²') => { self.pos += 1; op::SQR }
            Some('³') => { self.pos += 1; op::CUBE }
            Some('⁴') => { self.pos += 1; op::QUART }
            Some('^') => {
                self.pos += 1;
                self.skip_ws();
                let at = self.pos;
                let neg = self.eat('-');
                self.skip_ws();
                let start = self.pos;
                while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                    self.pos += 1;
                }
                let digits: String = self.chars[start..self.pos].iter().collect();
                match (neg, digits.as_str()) {
                    (false, "2") => op::SQR,
                    (false, "3") => op::CUBE,
                    (false, "4") => op::QUART,
                    (true, "1") => op::RECIP,
                    _ => return self.err(at, "only ^2, ^3, ^4 and ^-1 are supported"),
                }
            }
            _ => return Ok(a),
        };
        if matches!(self.peek(), Some('^' | '²' | '³' | '⁴')) {
            return self.err(self.pos, "chained powers need parentheses, e.g. (z^2)^2");
        }
        Ok(self.b.unary(o, a))
    }

    fn atom(&mut self) -> Result<usize, ParseError> {
        let start = match self.peek() {
            Some(_) => self.pos,
            None => return self.err(self.pos, "unexpected end of formula"),
        };
        let ch = self.chars[start];
        if ch == '(' {
            self.pos += 1;
            let e = self.expr()?;
            if !self.eat(')') {
                return self.err(self.pos, "expected ')'");
            }
            return Ok(e);
        }
        if ch.is_ascii_digit() || ch == '.' {
            while self.pos < self.chars.len() && (self.chars[self.pos].is_ascii_digit() || self.chars[self.pos] == '.') {
                self.pos += 1;
            }
            let s: String = self.chars[start..self.pos].iter().collect();
            let v: f32 = match s.parse() {
                Ok(v) => v,
                Err(_) => return self.err(start, format!("bad number '{s}'")),
            };
            // `0.5i` — an imaginary literal (no whitespace allowed before i).
            if self.chars.get(self.pos) == Some(&'i')
                && !self.chars.get(self.pos + 1).is_some_and(|c| c.is_ascii_alphanumeric())
            {
                self.pos += 1;
                return Ok(self.b.konst(0.0, v));
            }
            return Ok(self.b.konst(v, 0.0));
        }
        if ch.is_ascii_alphabetic() {
            while self.pos < self.chars.len() && (self.chars[self.pos].is_ascii_alphanumeric() || self.chars[self.pos] == '_') {
                self.pos += 1;
            }
            let name: String = self.chars[start..self.pos].iter().collect::<String>().to_lowercase();
            match name.as_str() {
                "z" => return Ok(self.b.push(OpNode { op: op::Z, a: 0, b: 0, kre: 0.0, kim: 0.0 })),
                "c" => return Ok(self.b.push(OpNode { op: op::C, a: 0, b: 0, kre: 0.0, kim: 0.0 })),
                "i" => return Ok(self.b.konst(0.0, 1.0)),
                "kbc" => {
                    // kbc(x, y): the constant (0, 0, x, y) in the B and C parts; x, y real numbers.
                    let real = |p: &mut Self| -> Result<f32, ParseError> {
                        let at = p.pos;
                        let e = p.expr()?;
                        match p.b.as_const(e) { Some((r, i)) if i == 0.0 => Ok(r), _ => p.err(at, "kbc(x, y) takes two real numbers") }
                    };
                    if !self.eat('(') { return self.err(self.pos, "expected '(' after kbc"); }
                    let x = real(self)?;
                    if !self.eat(',') { return self.err(self.pos, "expected ',' — kbc(x, y)"); }
                    let y = real(self)?;
                    if !self.eat(')') { return self.err(self.pos, "expected ')'"); }
                    return Ok(self.b.push(OpNode { op: op::CONSTBC, a: 0, b: 0, kre: x, kim: y }));
                }
                _ => {}
            }
            let Some(o) = func_op(&name) else {
                return self.err(start, format!("unknown name '{name}' (expected z, c, i or a function)"));
            };
            if !self.eat('(') {
                return self.err(self.pos, format!("expected '(' after {name}"));
            }
            let a = self.expr()?;
            if o == op::BULB {
                if !self.eat(',') {
                    return self.err(self.pos, "expected ',' — bulb(a, p) takes two arguments");
                }
                let p = self.expr()?;
                if !self.eat(')') {
                    return self.err(self.pos, "expected ')'");
                }
                return Ok(self.b.push(OpNode { op: op::BULB, a: a as u8, b: p as u8, kre: 0.0, kim: 0.0 }));
            }
            if !self.eat(')') {
                return self.err(self.pos, "expected ')'");
            }
            return Ok(self.b.unary(o, a));
        }
        self.err(start, format!("unexpected '{ch}'"))
    }
}

/// Keeps only the nodes `root` depends on, in a topological order that ends
/// with `root` — the layout the evaluator requires.
fn compact(nodes: &[OpNode], root: usize) -> Vec<OpNode> {
    fn visit(nodes: &[OpNode], i: usize, map: &mut Vec<Option<usize>>, out: &mut Vec<OpNode>) -> usize {
        if let Some(j) = map[i] {
            return j;
        }
        let mut n = nodes[i];
        let ar = op::arity(n.op);
        if ar >= 1 { n.a = visit(nodes, n.a as usize, map, out) as u8; }
        if ar >= 2 { n.b = visit(nodes, n.b as usize, map, out) as u8; }
        out.push(n);
        map[i] = Some(out.len() - 1);
        out.len() - 1
    }
    let mut map = vec![None; nodes.len()];
    let mut out = Vec::new();
    visit(nodes, root, &mut map, &mut out);
    out
}

/// Parses the text syntax described in the module docs into a program
/// (topological order, root last, at most `N_SLOTS` nodes).
pub fn parse_program(text: &str) -> Result<Vec<OpNode>, ParseError> {
    let mut p = Parser { chars: text.chars().collect(), pos: 0, b: Builder { nodes: Vec::new() } };
    let root = p.expr()?;
    if p.peek().is_some() {
        let c = p.chars[p.pos];
        return p.err(p.pos, format!("unexpected '{c}' (missing operator?)"));
    }
    let prog = compact(&p.b.nodes, root);
    if prog.len() > N_SLOTS {
        return Err(ParseError {
            pos: 0,
            msg: format!("formula needs {} nodes, the limit is {N_SLOTS}", prog.len()),
        });
    }
    Ok(prog)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quat_dag::eval_program_quat;
    use crate::quaternion::Quat;

    fn n(o: u8, a: u8, b: u8) -> OpNode { OpNode { op: o, a, b, kre: 0.0, kim: 0.0 } }
    fn k(re: f32, im: f32) -> OpNode { OpNode { op: op::CONST, a: 0, b: 0, kre: re, kim: im } }

    const SAMPLES: [(Quat, Quat); 4] = [
        (Quat::new(0.3, -0.2, 0.1, 0.05), Quat::new(-0.4, 0.6, 0.0, 0.2)),
        (Quat::new(-1.1, 0.4, -0.3, 0.7), Quat::new(0.25, 0.1, -0.5, 0.0)),
        (Quat::new(0.0, 0.0, 0.0, 0.0), Quat::new(0.1, 0.2, 0.3, 0.4)),
        (Quat::new(1.5, 1.0, -1.0, 0.5), Quat::new(-0.75, 0.0, 0.0, -0.3)),
    ];

    fn same(x: f64, y: f64) -> bool {
        (x.is_nan() && y.is_nan()) || x == y || (x - y).abs() <= 1e-12 * x.abs().max(y.abs())
    }

    fn assert_equivalent(a: &[OpNode], b: &[OpNode], ctx: &str) {
        for (z, c) in SAMPLES {
            let (x, y) = (eval_program_quat(a, z, c), eval_program_quat(b, z, c));
            assert!(
                same(x.r, y.r) && same(x.a, y.a) && same(x.b, y.b) && same(x.c, y.c),
                "{ctx}: {x:?} != {y:?}"
            );
        }
    }

    fn roundtrip(prog: &[OpNode]) -> Vec<OpNode> {
        let text = program_to_text(prog);
        parse_program(&text).unwrap_or_else(|e| panic!("reparse of {text:?} failed: {e}"))
    }

    #[test]
    fn every_op_round_trips() {
        for o in 0..op::N_OPS as u8 {
            // z, c, k(0.3,-0.7), then the op under test on (z, c) or the const.
            let mut prog = vec![n(op::Z, 0, 0), n(op::C, 0, 0), k(0.3, -0.7)];
            prog.push(match op::arity(o) {
                0 if o == op::CONST => k(-1.25, 0.5),
                0 if o == op::CONSTBC => OpNode { op: op::CONSTBC, a: 0, b: 0, kre: 0.4, kim: -0.9 },
                0 => n(o, 0, 0),
                1 => n(o, 0, 0),
                _ => n(o, 0, 1),
            });
            // Fold it back in with the constant so constants appear too.
            prog.push(n(op::MUL, 3, 2));
            assert_equivalent(&prog, &roundtrip(&prog), op::name(o));
        }
    }

    #[test]
    fn mandelbrot_text() {
        let p = parse_program("z^2 + c").unwrap();
        assert_eq!(p, vec![n(op::Z, 0, 0), n(op::SQR, 0, 0), n(op::C, 0, 0), n(op::ADD, 1, 2)]);
        assert_eq!(program_to_text(&p), "z^2 + c");
        assert_eq!(parse_program("z² + c").unwrap(), p);
    }

    #[test]
    fn operand_order_and_precedence_preserved() {
        // a - (b - c), a / (b * c), (a - b) * c: all need parens on print.
        let prog = vec![n(op::Z, 0, 0), n(op::C, 0, 0), n(op::SIN, 0, 0),
                        n(op::SUB, 1, 2), n(op::SUB, 0, 3), n(op::MUL, 1, 2), n(op::DIV, 4, 5),
                        n(op::SUB, 6, 0), n(op::MUL, 7, 1)];
        let text = program_to_text(&prog);
        assert_equivalent(&prog, &roundtrip(&prog), &text);
    }

    #[test]
    fn complex_and_negative_constants_are_exact() {
        for (re, im) in [(0.2f32, 0.1f32), (-0.2, -0.1), (0.0, -0.37), (-1.5, 0.0), (0.1, 0.0), (1e-7, 3.3e-5)] {
            let prog = vec![k(re, im), n(op::Z, 0, 0), n(op::MUL, 0, 1)];
            let back = roundtrip(&prog);
            let kc = back.iter().find(|m| m.op == op::CONST).unwrap();
            assert_eq!((kc.kre, kc.kim), (re, im), "text {}", program_to_text(&prog));
        }
    }

    #[test]
    fn unary_minus_on_expression() {
        let p = parse_program("-z^2 + c").unwrap();
        let expect = vec![k(0.0, 0.0), n(op::Z, 0, 0), n(op::SQR, 1, 0), n(op::SUB, 0, 2), n(op::C, 0, 0), n(op::ADD, 3, 4)];
        assert_equivalent(&p, &expect, "-z^2 + c");
    }

    #[test]
    fn shared_subexpressions_are_merged() {
        let p = parse_program("sin(z^2) * sin(z^2) + z^2").unwrap();
        assert_eq!(p.len(), 5, "{p:?}"); // z, z^2, sin, mul, add
    }

    #[test]
    fn errors_point_at_the_problem() {
        let e = parse_program("z^2 + foo(z)").unwrap_err();
        assert_eq!(e.pos, 6);
        let e = parse_program("z^5").unwrap_err();
        assert_eq!(e.pos, 2);
        assert!(parse_program("z^2 +").is_err());
        assert!(parse_program("(z + c").is_err());
        assert!(parse_program("z c").is_err());
        assert!(parse_program("z^2^2").is_err());
    }

    #[test]
    fn node_limit_enforced() {
        let long = (0..30).map(|i| format!("sin(z*{i}.5)")).collect::<Vec<_>>().join(" + ");
        let e = parse_program(&long).unwrap_err();
        assert!(e.msg.contains("limit"), "{e}");
    }

    #[test]
    fn eval_quirks_are_mirrored() {
        // Forward operand reference and unknown opcode both read zero.
        let prog = vec![n(op::Z, 0, 0), n(op::ADD, 0, 5), n(99, 0, 0), n(op::MUL, 1, 2), n(op::ADD, 3, 0)];
        assert_equivalent(&prog, &roundtrip(&prog), "quirks");
    }

    /// Every genome in the quat training corpus survives print → parse with
    /// an identical value. Skipped when the corpus is absent (fresh clone).
    #[test]
    fn corpus_genomes_round_trip() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("train_corpus_quat");
        let Ok(entries) = std::fs::read_dir(&dir) else { return };
        let mut checked = 0;
        for e in entries.flatten() {
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()) != Some("nn") { continue; }
            let Ok(g) = crate::io::load_genome(&path) else { continue };
            for (label, prog) in [("program", &g.program), ("warp", &g.warp)] {
                if prog.is_empty() { continue; }
                assert_equivalent(prog, &roundtrip(prog), &format!("{} {label}", path.display()));
                checked += 1;
            }
        }
        eprintln!("round-tripped {checked} programs");
    }
}
