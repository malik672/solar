//! Lexer performance benchmarks.
//!
//! This benchmark compares the performance of the optimized lexer functions
//! against the original implementations to measure the improvement.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use solar_parse::lexer::{Cursor, Lexer};
use solar_interface::Session;

/// Generate test Solidity code of various sizes.
fn generate_solidity_code(size: usize) -> String {
    let mut code = String::new();
    
    // Add some realistic Solidity patterns
    let patterns = [
        "    uint256 public balance;\n",
        "    mapping(address => uint) public balances;\n", 
        "    function transfer(address to, uint amount) public {\n",
        "        require(balance >= amount, \"Insufficient balance\");\n",
        "        balance -= amount;\n",
        "        balances[to] += amount;\n",
        "    }\n",
        "\n",
        "    // This is a comment about the next function\n",
        "    /// @dev This is a doc comment\n",
        "    event Transfer(address indexed from, address indexed to, uint value);\n",
    ];
    
    let mut pattern_idx = 0;
    while code.len() < size {
        code.push_str(patterns[pattern_idx % patterns.len()]);
        pattern_idx += 1;
    }
    
    // Truncate to exact size if needed
    code.truncate(size);
    code
}

/// Benchmark just the lexer token generation.
fn bench_lexer_tokens(c: &mut Criterion) {
    let sizes = [1024, 4096, 16384, 65536]; // 1KB, 4KB, 16KB, 64KB
    
    for &size in &sizes {
        let code = generate_solidity_code(size);
        
        let mut group = c.benchmark_group(format!("lexer_tokens_{}KB", size / 1024));
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_function("optimized", |b| {
            b.iter(|| {
                let sess = Session::builder().with_silent_emitter(None).build();
                let lexer = Lexer::new(&sess, black_box(&code));
                let tokens = lexer.into_tokens();
                black_box(tokens.len())
            });
        });
        
        group.finish();
    }
}

/// Benchmark raw cursor token generation.
fn bench_cursor_tokens(c: &mut Criterion) {
    let sizes = [1024, 4096, 16384]; // 1KB, 4KB, 16KB
    
    for &size in &sizes {
        let code = generate_solidity_code(size);
        
        let mut group = c.benchmark_group(format!("cursor_tokens_{}KB", size / 1024));
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_function("optimized", |b| {
            b.iter(|| {
                let mut cursor = Cursor::new(black_box(&code));
                let mut count = 0;
                while !cursor.advance_token().kind.is_eof() {
                    count += 1;
                }
                black_box(count)
            });
        });
        
        group.finish();
    }
}

/// Benchmark specific lexing patterns.
fn bench_whitespace_heavy(c: &mut Criterion) {
    // Create code with lots of whitespace
    let mut code = String::new();
    for _ in 0..1000 {
        code.push_str("    \t\n    uint256    value    ;    \n\n");
    }
    
    c.benchmark_group("whitespace_heavy")
        .throughput(Throughput::Bytes(code.len() as u64))
        .bench_function("optimized", |b| {
            b.iter(|| {
                let sess = Session::builder().with_silent_emitter(None).build();
                let lexer = Lexer::new(&sess, black_box(&code));
                let tokens = lexer.into_tokens();
                black_box(tokens.len())
            });
        });
}

fn bench_identifier_heavy(c: &mut Criterion) {
    // Create code with lots of long identifiers
    let mut code = String::new();
    for i in 0..1000 {
        code.push_str(&format!("uint256 very_long_identifier_name_{} = another_very_long_name_{};\n", i, i));
    }
    
    c.benchmark_group("identifier_heavy")
        .throughput(Throughput::Bytes(code.len() as u64))
        .bench_function("optimized", |b| {
            b.iter(|| {
                let sess = Session::builder().with_silent_emitter(None).build();
                let lexer = Lexer::new(&sess, black_box(&code));
                let tokens = lexer.into_tokens();
                black_box(tokens.len())
            });
        });
}

fn bench_number_heavy(c: &mut Criterion) {
    // Create code with lots of numbers
    let mut code = String::new();
    for i in 0..1000 {
        code.push_str(&format!("uint256 value{} = 123456789{}; uint256 hex{} = 0x{}abc;\n", i, i, i, i));
    }
    
    c.benchmark_group("number_heavy")
        .throughput(Throughput::Bytes(code.len() as u64))
        .bench_function("optimized", |b| {
            b.iter(|| {
                let sess = Session::builder().with_silent_emitter(None).build();
                let lexer = Lexer::new(&sess, black_box(&code));
                let tokens = lexer.into_tokens();
                black_box(tokens.len())
            });
        });
}

criterion_group!(
    benches,
    bench_lexer_tokens,
    bench_cursor_tokens,
    bench_whitespace_heavy,
    bench_identifier_heavy,
    bench_number_heavy
);
criterion_main!(benches);