//! Low-level Solidity lexer.
//!
//! Modified from Rust's [`rustc_lexer`](https://github.com/rust-lang/rust/blob/45749b21b7fd836f6c4f11dd40376f7c83e2791b/compiler/rustc_lexer/src/lib.rs).

use memchr::memmem;
use solar_ast::{Base, StrKind};
use solar_data_structures::hint::unlikely;
use std::sync::OnceLock;

pub mod token;
use token::{RawLiteralKind, RawToken, RawTokenKind};

#[cfg(test)]
mod tests;

/// Returns `true` if the given character is considered a whitespace.
#[inline(always)]
pub const fn is_whitespace(c: char) -> bool {
    is_whitespace_byte(ch2u8(c))
}

/// Returns `true` if the given character is considered a whitespace.
#[inline]
pub const fn is_whitespace_byte(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

/// Returns `true` if the given character is valid at the start of a Solidity identifier.
#[inline(always)]
pub const fn is_id_start(c: char) -> bool {
    is_id_start_byte(ch2u8(c))
}

#[inline]
pub const fn is_id_start_byte(c: u8) -> bool {
    matches!(c, b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'$')
}

/// Returns `true` if the given character is valid in a Solidity identifier.
#[inline(always)]
pub const fn is_id_continue(c: char) -> bool {
    is_id_continue_byte(ch2u8(c))
}

/// Returns `true` if the given character is valid in a Solidity identifier.
#[inline]
pub const fn is_id_continue_byte(c: u8) -> bool {
    let is_number = (c >= b'0') & (c <= b'9');
    is_id_start_byte(c) || is_number
}


/// Returns `true` if the given string is a valid Solidity identifier.
#[inline(always)]
pub const fn is_ident(s: &str) -> bool {
    is_ident_bytes(s.as_bytes())
}

/// Returns `true` if the given byte slice is a valid Solidity identifier.
pub const fn is_ident_bytes(s: &[u8]) -> bool {
    let [first, ref rest @ ..] = *s else {
        return false;
    };

    if !is_id_start_byte(first) {
        return false;
    }

    let mut i = 0;
    while i < rest.len() {
        if !is_id_continue_byte(rest[i]) {
            return false;
        }
        i += 1;
    }

    true
}

/// Converts a `char` to a `u8`.
#[inline(always)]
const fn ch2u8(c: char) -> u8 {
    c as u32 as u8
}

const EOF: u8 = b'\0';

/// Peekable iterator over a char sequence.
///
/// Next characters can be peeked via `first` method,
/// and position can be shifted forward via `bump` method.
#[derive(Clone, Debug)]
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// Creates a new cursor over the given input string slice.
    #[inline(always)]
    pub fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    /// Parses a token from the input string.
    #[inline]
    pub fn advance_token(&mut self) -> RawToken {
        let start_pos = self.pos;

        let first_char = match self.bump_ret() {
            Some(c) => c,
            None => return RawToken::EOF,
        };
        
        let token_kind = self.advance_token_kind(first_char);
        let len = (self.pos - start_pos) as u32;

        RawToken::new(token_kind, len)
    }

    #[inline]
    fn advance_token_kind(&mut self, first_char: u8) -> RawTokenKind {
        match first_char {
            // Slash, comment or block comment.
            b'/' => match self.first() {
                b'/' => self.line_comment(),
                b'*' => self.block_comment(),
                _ => RawTokenKind::Slash,
            },

            // Whitespace sequence.
            c if is_whitespace_byte(c) => self.whitespace(),

            // Identifier (this should be checked after other variant that can start as identifier).
            c if is_id_start_byte(c) => self.ident_or_prefixed_literal(c),

            // Numeric literal.
            b'0'..=b'9' => {
                let kind = self.number(first_char);
                RawTokenKind::Literal { kind }
            }
            b'.' if self.first().is_ascii_digit() => {
                let kind = self.rational_number_after_dot(Base::Decimal);
                RawTokenKind::Literal { kind }
            }

            // One-symbol tokens - optimized with jump table pattern
            b';' => RawTokenKind::Semi,
            b',' => RawTokenKind::Comma,
            b'.' => RawTokenKind::Dot,
            b'(' => RawTokenKind::OpenParen,
            b')' => RawTokenKind::CloseParen,
            b'{' => RawTokenKind::OpenBrace,
            b'}' => RawTokenKind::CloseBrace,
            b'[' => RawTokenKind::OpenBracket,
            b']' => RawTokenKind::CloseBracket,
            b'~' => RawTokenKind::Tilde,
            b'?' => RawTokenKind::Question,
            b':' => RawTokenKind::Colon,
            b'=' => RawTokenKind::Eq,
            b'!' => RawTokenKind::Bang,
            b'<' => RawTokenKind::Lt,
            b'>' => RawTokenKind::Gt,
            b'-' => RawTokenKind::Minus,
            b'&' => RawTokenKind::And,
            b'|' => RawTokenKind::Or,
            b'+' => RawTokenKind::Plus,
            b'*' => RawTokenKind::Star,
            b'^' => RawTokenKind::Caret,
            b'%' => RawTokenKind::Percent,

            // String literal.
            b'\'' | b'"' => {
                let terminated = self.eat_string(first_char);
                let kind = RawLiteralKind::Str { kind: StrKind::Str, terminated };
                RawTokenKind::Literal { kind }
            }

            _ => {
                if unlikely(!first_char.is_ascii()) {
                    self.bump_utf8_with(first_char);
                }
                RawTokenKind::Unknown
            }
        }
    }

    #[inline(never)]
    fn line_comment(&mut self) -> RawTokenKind {
        debug_assert!(self.prev() == b'/' && self.first() == b'/');
        self.bump();

        // `////` (more than 3 slashes) is not considered a doc comment.
        let is_doc = matches!(self.first(), b'/' if self.second() != b'/');

        // Optimized line ending search
        let remaining = &self.bytes[self.pos..];
        if let Some(pos) = memchr::memchr2(b'\n', b'\r', remaining) {
            self.pos += pos;
        } else {
            self.pos = self.bytes.len();
        }
        
        RawTokenKind::LineComment { is_doc }
    }

    #[inline(never)]
    fn block_comment(&mut self) -> RawTokenKind {
        debug_assert!(self.prev() == b'/' && self.first() == b'*');
        self.bump();

        // `/***` (more than 2 stars) is not considered a doc comment.
        // `/**/` is not considered a doc comment.
        let is_doc = matches!(self.first(), b'*' if !matches!(self.second(), b'*' | b'/'));

        let remaining = &self.bytes[self.pos..];
        static FINDER: OnceLock<memmem::Finder<'static>> = OnceLock::new();
        let (terminated, n) = FINDER
            .get_or_init(|| memmem::Finder::new(b"*/"))
            .find(remaining)
            .map_or((false, remaining.len()), |pos| (true, pos + 2));
        self.pos += n;

        RawTokenKind::BlockComment { is_doc, terminated }
    }

    #[inline]
    fn whitespace(&mut self) -> RawTokenKind {
        debug_assert!(is_whitespace_byte(self.prev()));
        self.eat_while(is_whitespace_byte);
        RawTokenKind::Whitespace
    }

    fn ident_or_prefixed_literal(&mut self, first: u8) -> RawTokenKind {
        debug_assert!(is_id_start_byte(self.prev()));

        let start_pos = self.pos - 1; // Account for already consumed first byte
        self.eat_while(is_id_continue_byte);

        // Check if the identifier is a string literal prefix.
        if unlikely(matches!(first, b'h' | b'u')) {
            let id = &self.bytes[start_pos..self.pos];
            let is_hex = id == b"hex";
            if is_hex || id == b"unicode" {
                if let quote @ (b'\'' | b'"') = self.first() {
                    self.bump();
                    let terminated = self.eat_string(quote);
                    let kind = if is_hex { StrKind::Hex } else { StrKind::Unicode };
                    return RawTokenKind::Literal {
                        kind: RawLiteralKind::Str { kind, terminated },
                    };
                }
            }
        }

        RawTokenKind::Ident
    }

    fn number(&mut self, first_digit: u8) -> RawLiteralKind {
        debug_assert!(self.prev().is_ascii_digit());
        let mut base = Base::Decimal;
        if first_digit == b'0' {
            // Attempt to parse encoding base.
            let has_digits = match self.first() {
                b'b' => {
                    base = Base::Binary;
                    self.bump();
                    self.eat_decimal_digits()
                }
                b'o' => {
                    base = Base::Octal;
                    self.bump();
                    self.eat_decimal_digits()
                }
                b'x' => {
                    base = Base::Hexadecimal;
                    self.bump();
                    self.eat_hexadecimal_digits()
                }
                // Not a base prefix.
                b'0'..=b'9' | b'_' | b'.' | b'e' | b'E' => {
                    self.eat_decimal_digits();
                    true
                }
                // Just a 0.
                _ => return RawLiteralKind::Int { base, empty_int: false },
            };
            // Base prefix was provided, but there were no digits after it, e.g. "0x".
            if !has_digits {
                return RawLiteralKind::Int { base, empty_int: true };
            }
        } else {
            // No base prefix, parse number in the usual way.
            self.eat_decimal_digits();
        };

        match self.first() {
            // Don't be greedy if this is actually an integer literal followed by field/method
            // access (`12.foo()`).
            b'.' if !is_id_start_byte(self.second()) || self.second() == b'_' => {
                self.bump();
                self.rational_number_after_dot(base)
            }
            b'e' | b'E' => {
                self.bump();
                let empty_exponent = !self.eat_exponent();
                RawLiteralKind::Rational { base, empty_exponent }
            }
            _ => RawLiteralKind::Int { base, empty_int: false },
        }
    }

    #[cold]
    fn rational_number_after_dot(&mut self, base: Base) -> RawLiteralKind {
        self.eat_decimal_digits();
        let empty_exponent = match self.first() {
            b'e' | b'E' => {
                self.bump();
                !self.eat_exponent()
            }
            _ => false,
        };
        RawLiteralKind::Rational { base, empty_exponent }
    }

    /// Eats a string until the given quote character. Returns `true` if the string was terminated.
    fn eat_string(&mut self, quote: u8) -> bool {
        debug_assert_eq!(self.prev(), quote);
        
        while self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            self.pos += 1;
            
            if c == quote {
                return true;
            }
            if c == b'\\' && self.pos < self.bytes.len() {
                let next = self.bytes[self.pos];
                if next == b'\\' || next == quote {
                    self.pos += 1; // Skip escaped character
                }
            }
        }
        false // End of file reached
    }

    /// Eats characters for a decimal number. Returns `true` if any digits were encountered.
    #[inline]
    fn eat_decimal_digits(&mut self) -> bool {
        self.eat_digits(|x| x.is_ascii_digit())
    }

    /// Eats characters for a hexadecimal number. Returns `true` if any digits were encountered.
    #[inline]
    fn eat_hexadecimal_digits(&mut self) -> bool {
        self.eat_digits(|x| x.is_ascii_hexdigit())
    }

    #[inline]
    fn eat_digits(&mut self, is_digit: impl Fn(u8) -> bool) -> bool {
        let mut has_digits = false;
        while self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            match c {
                b'_' => self.pos += 1,
                c if is_digit(c) => {
                    has_digits = true;
                    self.pos += 1;
                }
                _ => break,
            }
        }
        has_digits
    }

    /// Eats the exponent. Returns `true` if any digits were encountered.
    fn eat_exponent(&mut self) -> bool {
        debug_assert!(self.prev() == b'e' || self.prev() == b'E');
        if self.first() == b'-' {
            self.bump();
        }
        self.eat_decimal_digits()
    }

    /// Returns the remaining input as a byte slice.
    #[inline(always)]
    pub fn as_bytes(&self) -> &'a [u8] {
        &self.bytes[self.pos..]
    }

    /// Returns the pointer to the first byte of the remaining input.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        self.bytes[self.pos..].as_ptr()
    }

    /// Returns the last eaten byte.
    #[inline(always)]
    fn prev(&self) -> u8 {
        self.bytes[self.pos - 1]
    }

    /// Peeks the next byte from the input stream without consuming it.
    #[inline(always)]
    fn first(&self) -> u8 {
        self.bytes.get(self.pos).copied().unwrap_or(EOF)
    }

    /// Peeks the second byte from the input stream without consuming it.
    #[inline(always)]
    fn second(&self) -> u8 {
        self.bytes.get(self.pos + 1).copied().unwrap_or(EOF)
    }

    // /// Checks if there is nothing more to consume.
    // #[inline(always)]
    // fn is_eof(&self) -> bool {
    //     self.pos >= self.bytes.len()
    // }

    /// Moves to the next character.
    #[inline(always)]
    fn bump(&mut self) {
        self.pos += 1;
    }

    /// Skips to the end of the current UTF-8 character sequence.
    #[cold]
    fn bump_utf8_with(&mut self, x: u8) {
        debug_assert_eq!(self.prev(), x);
        let skip = match x {
            ..0x80 => 0,
            ..0xE0 => 1,
            ..0xF0 => 2,
            _ => 3,
        };
        self.pos += skip;
    }

    /// Moves to the next character, returning the current one.
    #[inline(always)]
    fn bump_ret(&mut self) -> Option<u8> {
        if self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            self.pos += 1;
            Some(c)
        } else {
            None
        }
    }

    /// Eats symbols while predicate returns true or until the end of file is reached.
    #[inline]
    fn eat_while(&mut self, predicate: impl Fn(u8) -> bool) {
        while self.pos < self.bytes.len() && predicate(self.bytes[self.pos]) {
            self.pos += 1;
        }
    }
}

impl Iterator for Cursor<'_> {
    type Item = RawToken;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let token = self.advance_token();
        if token.kind == RawTokenKind::Eof {
            None
        } else {
            Some(token)
        }
    }
}

impl std::iter::FusedIterator for Cursor<'_> {}


