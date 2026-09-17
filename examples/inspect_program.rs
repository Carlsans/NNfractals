use nnfractals::formula::op;

fn op_name(o: u8) -> &'static str {
    match o {
        op::Z => "Z", op::C => "C", op::CONST => "CONST",
        op::SQR => "SQR", op::CUBE => "CUBE", op::QUART => "QUART",
        op::RECIP => "RECIP", op::SIN => "SIN", op::COS => "COS",
        op::EXP => "EXP", op::LOG => "LOG", op::TANH => "TANH",
        op::CONJ => "CONJ", op::ABSFOLD => "ABSFOLD", op::ABSRE => "ABSRE",
        op::ABSIM => "ABSIM", op::NORMZ => "NORMZ", op::ADD => "ADD",
        op::SUB => "SUB", op::MUL => "MUL", op::DIV => "DIV",
        _ => "?",
    }
}

fn show(path: &str) {
    let g = nnfractals::io::load_genome(std::path::Path::new(path)).unwrap();
    println!("=== {path} ===");
    println!("julia_mode={} phoenix=({},{})", g.julia_mode, g.phoenix_re, g.phoenix_im);
    print!("prog: ");
    for (i, n) in g.program.iter().enumerate() {
        print!("[{i}]{}(a={},b={}) ", op_name(n.op), n.a, n.b);
    }
    println!();
    print!("warp: ");
    for (i, n) in g.warp.iter().enumerate() {
        print!("[{i}]{}(a={},b={}) ", op_name(n.op), n.a, n.b);
    }
    println!();
}

fn main() {
    for id in ["9aaae7276b77b52c", "a6598cb3b21c491b", "4ad359bb5cdcb646", "60e94f4f394bbdb2", "8e24fe3bf318e6e3"] {
        show(&format!("fractals_dag/{id}.nn"));
    }
}
