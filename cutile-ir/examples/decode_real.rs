fn main() {
    let path = std::env::args().nth(1).expect("path");
    let data = std::fs::read(&path).unwrap();
    let m = cutile_ir::bytecode::decoder::decode_module(&data).expect("decode");
    let b2 = cutile_ir::write_bytecode(&m).unwrap();
    let m2 = cutile_ir::bytecode::decoder::decode_module(&b2).expect("decode2");
    let b3 = cutile_ir::write_bytecode(&m2).unwrap();
    println!("orig {} -> reencode {} bytes", data.len(), b2.len());
    println!("idempotent (reencode==reencode2): {}", b2 == b3);
    println!("--- decoded module ---\n{m}");
}
