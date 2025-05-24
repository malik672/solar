//! Utilities for validating string and char literals and turning them into values they represent.

use alloy_primitives::hex;
use solar_data_structures::trustme;
use std::{borrow::Cow, ops::Range, slice, str::Chars};

mod errors;
pub(crate) use errors::emit_unescape_error;
pub use errors::EscapeError;

/// What kind of literal do we parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Normal string literal (e.g. `"a"`).
    Str,
    /// Unicode string literal (e.g. `unicode"😀"`).
    UnicodeStr,
    /// Hex string literal (e.g. `hex"1234"`).
    HexStr,
}

/// Parses a string literal (without quotes) into a byte array.
pub fn parse_string_literal(src: &str, mode: Mode) -> Cow<'_, [u8]> {
    try_parse_string_literal(src, mode, |_, _| {})
}

/// Parses a string literal (without quotes) into a byte array.
/// `f` is called for each escape error.
#[instrument(name = "parse_string_literal", level = "debug", skip_all)]
pub fn try_parse_string_literal<F>(src: &str, mode: Mode, f: F) -> Cow<'_, [u8]>
where
    F: FnMut(Range<usize>, EscapeError),
{
    let mut bytes = if needs_unescape(src, mode) {
        Cow::Owned(parse_literal_unescape(src, mode, f))
    } else {
        Cow::Borrowed(src.as_bytes())
    };
    if mode == Mode::HexStr {
        // Currently this should never fail, but it's a good idea to check anyway.
        if let Ok(decoded) = hex::decode(&bytes) {
            bytes = Cow::Owned(decoded);
        }
    }
    bytes
}

#[cold]
fn parse_literal_unescape<F>(src: &str, mode: Mode, f: F) -> Vec<u8>
where
    F: FnMut(Range<usize>, EscapeError),
{
    let mut bytes = Vec::with_capacity(src.len());
    parse_literal_unescape_into(src, mode, f, &mut bytes);
    bytes
}

fn parse_literal_unescape_into<F>(src: &str, mode: Mode, mut f: F, dst_buf: &mut Vec<u8>)
where
    F: FnMut(Range<usize>, EscapeError),
{
    // `src.len()` is enough capacity for the unescaped string, so we can just use a slice.
    // SAFETY: The buffer is never read from.
    debug_assert!(dst_buf.is_empty());
    debug_assert!(dst_buf.capacity() >= src.len());
    let mut dst = unsafe { slice::from_raw_parts_mut(dst_buf.as_mut_ptr(), dst_buf.capacity()) };
    unescape_literal_unchecked(src, mode, |range, res| match res {
        Ok(c) => {
            // NOTE: We can't use `char::encode_utf8` because `c` can be an invalid unicode code.
            let written = super::utf8::encode_utf8_raw(c, dst).len();

            // SAFETY: Unescaping guarantees that the final unescaped byte array is shorter than
            // the initial string.
            debug_assert!(dst.len() >= written);
            let advanced = unsafe { dst.get_unchecked_mut(written..) };

            // SAFETY: I don't know why this triggers E0521.
            dst = unsafe { trustme::decouple_lt_mut(advanced) };
        }
        Err(e) => f(range, e),
    });
    unsafe { dst_buf.set_len(dst_buf.capacity() - dst.len()) };
}

/// Unescapes the contents of a string literal (without quotes).
///
/// The callback is invoked with a range and either a unicode code point or an error.
#[instrument(level = "debug", skip_all)]
pub fn unescape_literal<F>(src: &str, mode: Mode, mut callback: F)
where
    F: FnMut(Range<usize>, Result<u32, EscapeError>),
{
    if needs_unescape(src, mode) {
        unescape_literal_unchecked(src, mode, callback)
    } else {
        for (i, ch) in src.char_indices() {
            callback(i..i + ch.len_utf8(), Ok(ch as u32));
        }
    }
}

/// Unescapes the contents of a string literal (without quotes).
///
/// See [`unescape_literal`] for more details.
fn unescape_literal_unchecked<F>(src: &str, mode: Mode, callback: F)
where
    F: FnMut(Range<usize>, Result<u32, EscapeError>),
{
    match mode {
        Mode::Str | Mode::UnicodeStr => {
            unescape_str(src, matches!(mode, Mode::UnicodeStr), callback)
        }
        Mode::HexStr => unescape_hex_str(src, callback),
    }
}

/// Fast-path check for whether a string literal needs to be unescaped or errors need to be
/// reported.
fn needs_unescape(src: &str, mode: Mode) -> bool {
    fn needs_unescape_chars(src: &str) -> bool {
        memchr::memchr3(b'\\', b'\n', b'\r', src.as_bytes()).is_some()
    }

    match mode {
        Mode::Str => needs_unescape_chars(src) || !src.is_ascii(),
        Mode::UnicodeStr => needs_unescape_chars(src),
        Mode::HexStr => src.len() % 2 != 0 || !hex::check_raw(src),
    }
}

fn scan_escape(chars: &mut Chars<'_>) -> Result<u32, EscapeError> {
    // Previous character was '\\', unescape what follows.
    // https://docs.soliditylang.org/en/latest/grammar.html#a4.SolidityLexer.EscapeSequence
    // Note that hex and unicode escape codes are not validated since string literals are allowed
    // to contain invalid UTF-8.
    Ok(match chars.next().ok_or(EscapeError::LoneSlash)? {
        // Both quotes are always valid escapes.
        '\'' => '\'' as u32,
        '"' => '"' as u32,

        '\\' => '\\' as u32,
        'n' => '\n' as u32,
        'r' => '\r' as u32,
        't' => '\t' as u32,

        'x' => {
            // Parse hexadecimal character code.
            let mut value = 0;
            for _ in 0..2 {
                let d = chars.next().ok_or(EscapeError::HexEscapeTooShort)?;
                let d = d.to_digit(16).ok_or(EscapeError::InvalidHexEscape)?;
                value = value * 16 + d;
            }
            value
        }

        'u' => {
            // Parse hexadecimal unicode character code.
            let mut value = 0;
            for _ in 0..4 {
                let d = chars.next().ok_or(EscapeError::UnicodeEscapeTooShort)?;
                let d = d.to_digit(16).ok_or(EscapeError::InvalidUnicodeEscape)?;
                value = value * 16 + d;
            }
            value
        }

        _ => return Err(EscapeError::InvalidEscape),
    })
}

/// Unescape characters in a string literal.
///
/// See [`unescape_literal`] for more details.
fn unescape_str<F>(src: &str, is_unicode: bool, mut callback: F)
where
    F: FnMut(Range<usize>, Result<u32, EscapeError>),
{
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    
    // Use unsafe indexing with manual bounds checking for better performance
    while i < len {
        let start = i;
        
        // SAFETY: i < len is checked above
        match unsafe { *bytes.get_unchecked(i) } {
            b'\\' => {
                // Check bounds once for escape sequence
                if i + 1 >= len {
                    callback(start..start + 1, Err(EscapeError::LoneSlash));
                    break;
                }
                
                // SAFETY: i + 1 < len is checked above
                match unsafe { *bytes.get_unchecked(i + 1) } {
                    // Line continuation: \r\n (check bounds once)
                    b'\r' if i + 2 < len && unsafe { *bytes.get_unchecked(i + 2) } == b'\n' => {
                        i = skip_ascii_whitespace_bytes_fast(bytes, i + 3, &mut callback);
                        continue;
                    },
                    // Line continuation: \n
                    b'\n' => {
                        i = skip_ascii_whitespace_bytes_fast(bytes, i + 2, &mut callback);
                        continue;
                    },
                    // Normal escape sequences
                    _ => {
                        let remaining = &src[i..];
                        let mut chars = remaining.chars();
                        chars.next(); // Skip backslash
                        
                        let res = scan_escape(&mut chars);
                        let consumed = remaining.len() - chars.as_str().len();
                        callback(start..start + consumed, res);
                        i += consumed;
                    }
                }
            },
            b'\n' => {
                callback(start..start + 1, Err(EscapeError::StrNewline));
                i += 1;
            },
            b'\r' => {
                // Check for \r\n sequence
                if i + 1 < len && unsafe { *bytes.get_unchecked(i + 1) } == b'\n' {
                    i += 1; // Skip \r, will process \n next
                    continue;
                }
                callback(start..start + 1, Err(EscapeError::BareCarriageReturn));
                i += 1;
            },
            // Fast path for ASCII (0-127) - most common case
            ascii_byte @ 0..=127 => {
                // Batch process ASCII characters for better performance
                let ascii_end = find_next_non_ascii_or_special(bytes, i, len);
                if ascii_end > i + 1 {
                    // Process multiple ASCII characters at once
                    for j in i..ascii_end {
                        let byte = unsafe { *bytes.get_unchecked(j) };
                        callback(j..j + 1, Ok(byte as u32));
                    }
                    i = ascii_end;
                } else {
                    callback(start..start + 1, Ok(ascii_byte as u32));
                    i += 1;
                }
            },
            // Non-ASCII UTF-8 sequences
            _ => {
                // Optimized UTF-8 handling with better error checking
                let (ch_len, codepoint) = decode_utf8_with_len(&bytes[i..]);
                let actual_len = ch_len.min(len - i);
                
                if !is_unicode {
                    callback(start..start + actual_len, Err(EscapeError::StrNonAsciiChar));
                } else {
                    callback(start..start + actual_len, Ok(codepoint));
                }
                i += actual_len;
            }
        }
    }
}

// Optimized function to find the end of a run of ASCII characters
#[inline]
fn find_next_non_ascii_or_special(bytes: &[u8], start: usize, len: usize) -> usize {
    let mut i = start;
    while i < len {
        match unsafe { *bytes.get_unchecked(i) } {
            // Stop at special characters that need processing
            b'\\' | b'\n' | b'\r' => break,
            // Stop at non-ASCII
            128..=255 => break,
            // Continue for regular ASCII
            _ => i += 1,
        }
    }
    i
}

// Combined UTF-8 length detection and decoding for better performance
#[inline]
fn decode_utf8_with_len(bytes: &[u8]) -> (usize, u32) {
    if bytes.is_empty() {
        return (0, 0xFFFD);
    }
    
    let first = bytes[0];
    match first {
        // ASCII - fast path
        0x00..=0x7F => (1, first as u32),
        
        // 2-byte sequence
        0xC0..=0xDF => {
            if bytes.len() >= 2 {
                let b1 = first as u32;
                let b2 = bytes[1] as u32;
                // Basic validation
                if (b2 & 0xC0) == 0x80 {
                    (2, ((b1 & 0x1F) << 6) | (b2 & 0x3F))
                } else {
                    (1, 0xFFFD)
                }
            } else {
                (1, 0xFFFD)
            }
        },
        
        // 3-byte sequence
        0xE0..=0xEF => {
            if bytes.len() >= 3 {
                let b1 = first as u32;
                let b2 = bytes[1] as u32;
                let b3 = bytes[2] as u32;
                // Basic validation
                if (b2 & 0xC0) == 0x80 && (b3 & 0xC0) == 0x80 {
                    (3, ((b1 & 0x0F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F))
                } else {
                    (1, 0xFFFD)
                }
            } else {
                (1, 0xFFFD)
            }
        },
        
        // 4-byte sequence
        0xF0..=0xF7 => {
            if bytes.len() >= 4 {
                let b1 = first as u32;
                let b2 = bytes[1] as u32;
                let b3 = bytes[2] as u32;
                let b4 = bytes[3] as u32;
                // Basic validation
                if (b2 & 0xC0) == 0x80 && (b3 & 0xC0) == 0x80 && (b4 & 0xC0) == 0x80 {
                    (4, ((b1 & 0x07) << 18) | ((b2 & 0x3F) << 12) | ((b3 & 0x3F) << 6) | (b4 & 0x3F))
                } else {
                    (1, 0xFFFD)
                }
            } else {
                (1, 0xFFFD)
            }
        },
        
        // Invalid UTF-8 start byte
        _ => (1, 0xFFFD),
    }
}

// Optimized whitespace skipping with fewer bounds checks
fn skip_ascii_whitespace_bytes_fast<F>(bytes: &[u8], mut i: usize, callback: &mut F) -> usize 
where
    F: FnMut(Range<usize>, Result<u32, EscapeError>),
{
    let len = bytes.len();
    
    while i < len {
        // SAFETY: i < len is checked above
        match unsafe { *bytes.get_unchecked(i) } {
            b' ' | b'\t' => {
                // Skip consecutive spaces/tabs in one go
                while i < len {
                    match unsafe { *bytes.get_unchecked(i) } {
                        b' ' | b'\t' => i += 1,
                        _ => break,
                    }
                }
            },
            b'\n' => {
                callback(i..i + 1, Err(EscapeError::CannotSkipMultipleLines));
                i += 1;
            },
            b'\r' => {
                // Check for \r\n
                if i + 1 < len && unsafe { *bytes.get_unchecked(i + 1) } == b'\n' {
                    callback(i..i + 2, Err(EscapeError::CannotSkipMultipleLines));
                    i += 2;
                } else {
                    callback(i..i + 1, Err(EscapeError::BareCarriageReturn));
                    i += 1;
                }
            },
            _ => break,
        }
    }
    i
}




/// Unescape characters in a hex string literal.
///
/// See [`unescape_literal`] for more details.
fn unescape_hex_str<F>(src: &str, mut callback: F)
where
    F: FnMut(Range<usize>, Result<u32, EscapeError>),
{
    let mut chars = src.char_indices();
    if src.starts_with("0x") || src.starts_with("0X") {
        chars.next();
        chars.next();
        callback(0..2, Err(EscapeError::HexPrefix));
    }

    let count = chars.clone().filter(|(_, c)| c.is_ascii_hexdigit()).count();
    if count % 2 != 0 {
        callback(0..src.len(), Err(EscapeError::HexOddDigits));
        return;
    }

    let mut emit_underscore_errors = true;
    let mut allow_underscore = false;
    let mut even = true;
    for (start, c) in chars {
        let res = match c {
            '_' => {
                if emit_underscore_errors && (!allow_underscore || !even) {
                    // Don't spam errors for multiple underscores.
                    emit_underscore_errors = false;
                    Err(EscapeError::HexBadUnderscore)
                } else {
                    allow_underscore = false;
                    continue;
                }
            }
            c if !c.is_ascii_hexdigit() => Err(EscapeError::HexNotHexDigit),
            c => Ok(c as u32),
        };

        if res.is_ok() {
            even = !even;
            allow_underscore = true;
        }

        let end = start + c.len_utf8();
        callback(start..end, res);
    }

    if emit_underscore_errors && src.len() > 1 && src.ends_with('_') {
        callback(src.len() - 1..src.len(), Err(EscapeError::HexBadUnderscore));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use EscapeError::*;

    type ExErr = (Range<usize>, EscapeError);

    fn check(mode: Mode, src: &str, expected_str: &str, expected_errs: &[ExErr]) {
        let panic_str = format!("{mode:?}: {src:?}");

        let mut ok = String::with_capacity(src.len());
        let mut errs = Vec::with_capacity(expected_errs.len());
        unescape_literal(src, mode, |range, c| match c {
            Ok(c) => ok.push(char::try_from(c).unwrap()),
            Err(e) => errs.push((range, e)),
        });
        assert_eq!(errs, expected_errs, "{panic_str}");
        assert_eq!(ok, expected_str, "{panic_str}");

        let mut errs2 = Vec::with_capacity(errs.len());
        let out = try_parse_string_literal(src, mode, |range, e| {
            errs2.push((range, e));
        });
        assert_eq!(errs2, errs, "{panic_str}");
        if mode == Mode::HexStr {
            assert_eq!(hex::encode(out), expected_str, "{panic_str}");
        } else {
            assert_eq!(hex::encode(out), hex::encode(expected_str), "{panic_str}");
        }
    }

    #[test]
    fn unescape_str() {
        let cases: &[(&str, &str, &[ExErr])] = &[
            ("", "", &[]),
            (" ", " ", &[]),
            ("\t", "\t", &[]),
            (" \t ", " \t ", &[]),
            ("foo", "foo", &[]),
            ("hello world", "hello world", &[]),
            (r"\", "", &[(0..1, LoneSlash)]),
            (r"\\", "\\", &[]),
            (r"\\\", "\\", &[(2..3, LoneSlash)]),
            (r"\\\\", "\\\\", &[]),
            (r"\\ ", "\\ ", &[]),
            (r"\\ \", "\\ ", &[(3..4, LoneSlash)]),
            (r"\\ \\", "\\ \\", &[]),
            (r"\x", "", &[(0..2, HexEscapeTooShort)]),
            (r"\x1", "", &[(0..3, HexEscapeTooShort)]),
            (r"\xz", "", &[(0..3, InvalidHexEscape)]),
            (r"\xzf", "f", &[(0..3, InvalidHexEscape)]),
            (r"\xzz", "z", &[(0..3, InvalidHexEscape)]),
            (r"\x69", "\x69", &[]),
            (r"\xE8", "è", &[]),
            (r"\u", "", &[(0..2, UnicodeEscapeTooShort)]),
            (r"\u1", "", &[(0..3, UnicodeEscapeTooShort)]),
            (r"\uz", "", &[(0..3, InvalidUnicodeEscape)]),
            (r"\uzf", "f", &[(0..3, InvalidUnicodeEscape)]),
            (r"\u12", "", &[(0..4, UnicodeEscapeTooShort)]),
            (r"\u123", "", &[(0..5, UnicodeEscapeTooShort)]),
            (r"\u1234", "\u{1234}", &[]),
            (r"\u00e8", "è", &[]),
            (r"\r", "\r", &[]),
            (r"\t", "\t", &[]),
            (r"\n", "\n", &[]),
            (r"\n\n", "\n\n", &[]),
            (r"\ ", "", &[(0..2, InvalidEscape)]),
            (r"\?", "", &[(0..2, InvalidEscape)]),
            ("\r\n", "", &[(1..2, StrNewline)]),
            ("\n", "", &[(0..1, StrNewline)]),
            ("\\\n", "", &[]),
            ("\\\na", "a", &[]),
            ("\\\n  a", "a", &[]),
            ("a \\\n  b", "a b", &[]),
            ("a\\n\\\n  b", "a\nb", &[]),
            ("a\\t\\\n  b", "a\tb", &[]),
            ("a\\n \\\n  b", "a\n b", &[]),
            ("a\\n \\\n \tb", "a\n b", &[]),
            ("a\\t \\\n  b", "a\t b", &[]),
            ("\\\n \t a", "a", &[]),
            (" \\\n \t a", " a", &[]),
            ("\\\n \t a\n", "a", &[(6..7, StrNewline)]),
            ("\\\n   \t   ", "", &[]),
            (" \\\n   \t   ", " ", &[]),
            (" he\\\n \\\nllo \\\n wor\\\nld", " hello world", &[]),
            ("\\\n\na\\\nb", "ab", &[(2..3, CannotSkipMultipleLines)]),
            ("\\\n \na\\\nb", "ab", &[(3..4, CannotSkipMultipleLines)]),
            (
                "\\\n \n\na\\\nb",
                "ab",
                &[(3..4, CannotSkipMultipleLines), (4..5, CannotSkipMultipleLines)],
            ),
            (
                "a\\\n \n \t \nb\\\nc",
                "abc",
                &[(4..5, CannotSkipMultipleLines), (8..9, CannotSkipMultipleLines)],
            ),
        ];
        for &(src, expected_str, expected_errs) in cases {
            check(Mode::Str, src, expected_str, expected_errs);
            check(Mode::UnicodeStr, src, expected_str, expected_errs);
        }
    }

    #[test]
    fn unescape_unicode_str() {
        let cases: &[(&str, &str, &[ExErr], &[ExErr])] = &[
            ("è", "è", &[], &[(0..2, StrNonAsciiChar)]),
            ("😀", "😀", &[], &[(0..4, StrNonAsciiChar)]),
        ];
        for &(src, expected_str, e1, e2) in cases {
            check(Mode::UnicodeStr, src, expected_str, e1);
            check(Mode::Str, src, "", e2);
        }
    }

    #[test]
    fn unescape_hex_str() {
        let cases: &[(&str, &str, &[ExErr])] = &[
            ("", "", &[]),
            ("z", "", &[(0..1, HexNotHexDigit)]),
            ("\n", "", &[(0..1, HexNotHexDigit)]),
            ("  11", "11", &[(0..1, HexNotHexDigit), (1..2, HexNotHexDigit)]),
            ("0x", "", &[(0..2, HexPrefix)]),
            ("0X", "", &[(0..2, HexPrefix)]),
            ("0x11", "11", &[(0..2, HexPrefix)]),
            ("0X11", "11", &[(0..2, HexPrefix)]),
            ("1", "", &[(0..1, HexOddDigits)]),
            ("12", "12", &[]),
            ("123", "", &[(0..3, HexOddDigits)]),
            ("1234", "1234", &[]),
            ("_", "", &[(0..1, HexBadUnderscore)]),
            ("_11", "11", &[(0..1, HexBadUnderscore)]),
            ("_11_", "11", &[(0..1, HexBadUnderscore)]),
            ("11_", "11", &[(2..3, HexBadUnderscore)]),
            ("11_22", "1122", &[]),
            ("11__", "11", &[(3..4, HexBadUnderscore)]),
            ("11__22", "1122", &[(3..4, HexBadUnderscore)]),
            ("1_2", "12", &[(1..2, HexBadUnderscore)]),
        ];
        for &(src, expected_str, expected_errs) in cases {
            check(Mode::HexStr, src, expected_str, expected_errs);
        }
    }
}
