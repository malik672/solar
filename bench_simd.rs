use std::time::Instant;
use solar_parse::lexer::simd_lexer::*;

fn benchmark_function<F>(name: &str, func: F, input: &[u8], iterations: usize) -> u64
where 
    F: Fn(&[u8]) -> usize,
{
    // Warmup
    for _ in 0..100 {
        std::hint::black_box(func(input));
    }
    
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(func(input));
    }
    let elapsed = start.elapsed();
    
    let ns_per_op = elapsed.as_nanos() as u64 / iterations as u64;
    println!("{}: {} ns/op ({} ops in {:?})", name, ns_per_op, iterations, elapsed);
    ns_per_op
}

// Old branchy implementation for comparison
fn skip_whitespace_branchy(input: &[u8]) -> usize {
    let mut pos = 0;
    
    while pos + 8 <= input.len() {
        let chunk = &input[pos..pos + 8];
        
        let mut count = 0;
        for &byte in chunk {
            if matches!(byte, b' ' | b'\t' | b'\n' | b'\r') {
                count += 1;
            } else {
                return pos + count;
            }
        }
        
        if count == 8 {
            pos += 8;
        } else {
            return pos + count;
        }
    }
    
    while pos < input.len() && matches!(input[pos], b' ' | b'\t' | b'\n' | b'\r') {
        pos += 1;
    }
    
    pos
}

fn parse_identifier_branchy(input: &[u8]) -> usize {
    let mut pos = 0;
    
    while pos + 8 <= input.len() {
        let chunk = &input[pos..pos + 8];
        
        let mut count = 0;
        for &byte in chunk {
            let is_continue = matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'$' | b'0'..=b'9');
            if is_continue {
                count += 1;
            } else {
                return pos + count;
            }
        }
        
        if count == 8 {
            pos += 8;
        } else {
            return pos + count;
        }
    }
    
    while pos < input.len() {
        let byte = input[pos];
        let is_continue = matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'$' | b'0'..=b'9');
        if is_continue {
            pos += 1;
        } else {
            break;
        }
    }
    
    pos
}

fn main() {
    println!("SIMD Branchless vs Branchy Implementation Benchmark\n");
    
    // Test cases of varying sizes and patterns
    let test_cases = [
        ("small_whitespace", "    \t\n  hello".as_bytes()),
        ("medium_whitespace", format!("{}code", " ".repeat(50)).as_bytes()),
        ("large_whitespace", format!("{}code", " ".repeat(200)).as_bytes()),
        ("small_identifier", "myVariable123_$ ".as_bytes()),
        ("medium_identifier", format!("{}_ ", "a".repeat(50)).as_bytes()),
        ("large_identifier", format!("{}_ ", "myVeryLongIdentifierName".repeat(10)).as_bytes()),
        ("mixed_content", "   identifier123   more_stuff_here  ".as_bytes()),
        ("no_match", "!@#$%^&*()".as_bytes()),
    ];
    
    let iterations = 100_000;
    
    for (name, input) in &test_cases {
        println!("\n=== {} ({} bytes) ===", name, input.len());
        
        // Benchmark whitespace functions
        let branchy_ws = benchmark_function(
            "Whitespace (branchy)", 
            skip_whitespace_branchy, 
            input, 
            iterations
        );
        
        let branchless_ws = benchmark_function(
            "Whitespace (branchless)", 
            skip_whitespace_bulk, 
            input, 
            iterations
        );
        
        let ws_improvement = if branchless_ws < branchy_ws {
            format!("{:.1}% faster", ((branchy_ws - branchless_ws) as f64 / branchy_ws as f64) * 100.0)
        } else {
            format!("{:.1}% slower", ((branchless_ws - branchy_ws) as f64 / branchy_ws as f64) * 100.0)
        };
        
        // Benchmark identifier functions  
        let branchy_id = benchmark_function(
            "Identifier (branchy)", 
            parse_identifier_branchy, 
            input, 
            iterations
        );
        
        let branchless_id = benchmark_function(
            "Identifier (branchless)", 
            parse_identifier_bulk, 
            input, 
            iterations
        );
        
        let id_improvement = if branchless_id < branchy_id {
            format!("{:.1}% faster", ((branchy_id - branchless_id) as f64 / branchy_id as f64) * 100.0)
        } else {
            format!("{:.1}% slower", ((branchless_id - branchy_id) as f64 / branchy_id as f64) * 100.0)
        };
        
        println!("Whitespace: {}", ws_improvement);
        println!("Identifier: {}", id_improvement);
        
        // Verify correctness
        assert_eq!(skip_whitespace_branchy(input), skip_whitespace_bulk(input), 
                   "Whitespace results differ for {}", name);
        assert_eq!(parse_identifier_branchy(input), parse_identifier_bulk(input), 
                   "Identifier results differ for {}", name);
    }
    
    println!("\n✅ All implementations produce identical results!");
}