use nnfractals::formula::op;
use std::collections::HashMap;

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

fn main() {
    let mut files: Vec<_> = std::fs::read_dir("fractals_dag_quat").unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nn"))
        .collect();
    files.sort();
    let mut program_shapes: HashMap<String, usize> = HashMap::new();
    for path in &files {
        let g = nnfractals::io::load_genome(path).unwrap();
        let shape: Vec<&str> = g.program.iter().map(|n| op_name(n.op)).collect();
        let key = shape.join(",");
        *program_shapes.entry(key.clone()).or_insert(0) += 1;
        println!("{}: [{}]  (nodes={})", path.file_stem().unwrap().to_str().unwrap(), key, g.program.len());
    }
    println!("\n{} unique program shapes among {} genomes", program_shapes.len(), files.len());
}
