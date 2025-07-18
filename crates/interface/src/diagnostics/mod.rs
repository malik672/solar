//! Diagnostics implementation.
//!
//! Modified from [`rustc_errors`](https://github.com/rust-lang/rust/blob/520e30be83b4ed57b609d33166c988d1512bf4f3/compiler/rustc_errors/src/diagnostic.rs).

use crate::Span;
use anstyle::{AnsiColor, Color};
use std::{
    borrow::Cow,
    fmt::{self, Write},
    panic::Location,
};

mod builder;
pub use builder::{DiagBuilder, EmissionGuarantee};

mod context;
pub use context::{DiagCtxt, DiagCtxtFlags};

mod emitter;
#[cfg(feature = "json")]
pub use emitter::JsonEmitter;
pub use emitter::{
    DynEmitter, Emitter, HumanBufferEmitter, HumanEmitter, LocalEmitter, SilentEmitter,
};

mod message;
pub use message::{DiagMsg, MultiSpan, SpanLabel};

/// Represents all the diagnostics emitted up to a certain point.
///
/// Returned by [`DiagCtxt::emitted_diagnostics`].
pub struct EmittedDiagnostics(pub(crate) String);

impl fmt::Debug for EmittedDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Display for EmittedDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EmittedDiagnostics {}

impl EmittedDiagnostics {
    /// Returns `true` if no diagnostics have been emitted.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Useful type to use with [`Result`] indicate that an error has already been reported to the user,
/// so no need to continue checking.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ErrorGuaranteed(());

impl fmt::Debug for ErrorGuaranteed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ErrorGuaranteed")
    }
}

impl ErrorGuaranteed {
    /// Creates a new `ErrorGuaranteed`.
    ///
    /// Use of this method is discouraged.
    #[inline]
    pub const fn new_unchecked() -> Self {
        Self(())
    }
}

/// Marker type which enables implementation of `create_bug` and `emit_bug` functions for
/// bug diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BugAbort;

/// Signifies that the compiler died with an explicit call to `.bug` rather than a failed assertion,
/// etc.
pub struct ExplicitBug;

/// Marker type which enables implementation of fatal diagnostics.
pub struct FatalAbort;

/// Diag ID.
///
/// Use [`error_code!`](crate::error_code) to create an error code diagnostic ID.
///
/// # Examples
///
/// ```
/// # use solar_interface::{diagnostics::DiagId, error_code};
/// let id: DiagId = error_code!(1234);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DiagId {
    s: Cow<'static, str>,
}

impl DiagId {
    /// Creates a new diagnostic ID from a number.
    ///
    /// This should be used for custom lints. For solc-like error codes, use
    /// the [`error_code!`](crate::error_code) macro.
    pub fn new_str(s: impl Into<Cow<'static, str>>) -> Self {
        Self { s: s.into() }
    }

    /// Creates an error code diagnostic ID.
    ///
    /// Use [`error_code!`](crate::error_code) instead.
    #[doc(hidden)]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn new_from_macro(id: u32) -> Self {
        debug_assert!((1..=9999).contains(&id), "error code must be in range 0001-9999");
        Self { s: Cow::Owned(format!("{id:04}")) }
    }

    /// Returns the string representation of the diagnostic ID.
    pub fn as_string(&self) -> String {
        self.s.to_string()
    }
}

/// Used for creating an error code. The input must be exactly 4 decimal digits.
///
/// # Examples
///
/// ```
/// # use solar_interface::{diagnostics::DiagId, error_code};
/// let code: DiagId = error_code!(1234);
/// ```
#[macro_export]
macro_rules! error_code {
    ($id:literal) => {
        $crate::diagnostics::DiagId::new_from_macro($id)
    };
}

/// Diag level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    /// For bugs in the compiler. Manifests as an ICE (internal compiler error) panic.
    ///
    /// Its `EmissionGuarantee` is `BugAbort`.
    Bug,

    /// An error that causes an immediate abort. Used for things like configuration errors,
    /// internal overflows, some file operation errors.
    ///
    /// Its `EmissionGuarantee` is `FatalAbort`.
    Fatal,

    /// An error in the code being compiled, which prevents compilation from finishing. This is the
    /// most common case.
    ///
    /// Its `EmissionGuarantee` is `ErrorGuaranteed`.
    Error,

    /// A warning about the code being compiled. Does not prevent compilation from finishing.
    ///
    /// Its `EmissionGuarantee` is `()`.
    Warning,

    /// A message giving additional context. Rare, because notes are more commonly attached to
    /// other diagnostics such as errors.
    ///
    /// Its `EmissionGuarantee` is `()`.
    Note,

    /// A note that is only emitted once. Rare, mostly used in circumstances relating to lints.
    ///
    /// Its `EmissionGuarantee` is `()`.
    OnceNote,

    /// A message suggesting how to fix something. Rare, because help messages are more commonly
    /// attached to other diagnostics such as errors.
    ///
    /// Its `EmissionGuarantee` is `()`.
    Help,

    /// A help that is only emitted once. Rare.
    ///
    /// Its `EmissionGuarantee` is `()`.
    OnceHelp,

    /// Similar to `Note`, but used in cases where compilation has failed. Rare.
    ///
    /// Its `EmissionGuarantee` is `()`.
    FailureNote,

    /// Only used for lints.
    ///
    /// Its `EmissionGuarantee` is `()`.
    Allow,
}

impl Level {
    /// Returns the string representation of the level.
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Bug => "error: internal compiler error",
            Self::Fatal | Self::Error => "error",
            Self::Warning => "warning",
            Self::Note | Self::OnceNote => "note",
            Self::Help | Self::OnceHelp => "help",
            Self::FailureNote => "failure-note",
            Self::Allow
            // | Self::Expect(_)
            => unreachable!(),
        }
    }

    /// Returns `true` if this level is an error.
    #[inline]
    pub fn is_error(self) -> bool {
        match self {
            Self::Bug | Self::Fatal | Self::Error | Self::FailureNote => true,

            Self::Warning
            | Self::Note
            | Self::OnceNote
            | Self::Help
            | Self::OnceHelp
            | Self::Allow => false,
        }
    }

    /// Returns the style of this level.
    #[inline]
    pub const fn style(self) -> anstyle::Style {
        anstyle::Style::new().fg_color(self.color()).bold()
    }

    /// Returns the color of this level.
    #[inline]
    pub const fn color(self) -> Option<Color> {
        match self.ansi_color() {
            Some(c) => Some(Color::Ansi(c)),
            None => None,
        }
    }

    /// Returns the ANSI color of this level.
    #[inline]
    pub const fn ansi_color(self) -> Option<AnsiColor> {
        // https://github.com/rust-lang/rust/blob/99472c7049783605444ab888a97059d0cce93a12/compiler/rustc_errors/src/lib.rs#L1768
        match self {
            Self::Bug | Self::Fatal | Self::Error => Some(AnsiColor::BrightRed),
            Self::Warning => Some(AnsiColor::BrightYellow),
            Self::Note | Self::OnceNote => Some(AnsiColor::BrightGreen),
            Self::Help | Self::OnceHelp => Some(AnsiColor::BrightCyan),
            Self::FailureNote | Self::Allow => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Style {
    MainHeaderMsg,
    HeaderMsg,
    LineAndColumn,
    LineNumber,
    Quotation,
    UnderlinePrimary,
    UnderlineSecondary,
    LabelPrimary,
    LabelSecondary,
    NoStyle,
    Level(Level),
    Highlight,
    Addition,
    Removal,
}

impl Style {
    /// Converts the style to an [`anstyle::Style`].
    pub const fn to_color_spec(self, level: Level) -> anstyle::Style {
        use AnsiColor::*;

        /// On Windows, BRIGHT_BLUE is hard to read on black. Use cyan instead.
        ///
        /// See [rust-lang/rust#36178](https://github.com/rust-lang/rust/pull/36178).
        const BRIGHT_BLUE: Color = Color::Ansi(if cfg!(windows) { BrightCyan } else { BrightBlue });
        const GREEN: Color = Color::Ansi(BrightGreen);
        const MAGENTA: Color = Color::Ansi(BrightMagenta);
        const RED: Color = Color::Ansi(BrightRed);
        const WHITE: Color = Color::Ansi(BrightWhite);

        let s = anstyle::Style::new();
        match self {
            Self::Addition => s.fg_color(Some(GREEN)),
            Self::Removal => s.fg_color(Some(RED)),
            Self::LineAndColumn => s,
            Self::LineNumber => s.fg_color(Some(BRIGHT_BLUE)).bold(),
            Self::Quotation => s,
            Self::MainHeaderMsg => if cfg!(windows) { s.fg_color(Some(WHITE)) } else { s }.bold(),
            Self::UnderlinePrimary | Self::LabelPrimary => s.fg_color(level.color()).bold(),
            Self::UnderlineSecondary | Self::LabelSecondary => s.fg_color(Some(BRIGHT_BLUE)).bold(),
            Self::HeaderMsg | Self::NoStyle => s,
            Self::Level(level2) => s.fg_color(level2.color()).bold(),
            Self::Highlight => s.fg_color(Some(MAGENTA)).bold(),
        }
    }
}

/// A "sub"-diagnostic attached to a parent diagnostic.
/// For example, a note attached to an error.
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct SubDiagnostic {
    pub level: Level,
    pub messages: Vec<(DiagMsg, Style)>,
    pub span: MultiSpan,
}

impl SubDiagnostic {
    /// Formats the diagnostic messages into a single string.
    pub fn label(&self) -> Cow<'_, str> {
        flatten_messages(&self.messages, false, self.level)
    }

    /// Formats the diagnostic messages into a single string with ANSI color codes if applicable.
    pub fn label_with_style(&self, supports_color: bool) -> Cow<'_, str> {
        flatten_messages(&self.messages, supports_color, self.level)
    }
}

/// A compiler diagnostic.
#[must_use]
#[derive(Clone, Debug)]
pub struct Diag {
    pub(crate) level: Level,

    pub messages: Vec<(DiagMsg, Style)>,
    pub span: MultiSpan,
    pub children: Vec<SubDiagnostic>,
    pub code: Option<DiagId>,

    pub created_at: &'static Location<'static>,
}

impl PartialEq for Diag {
    fn eq(&self, other: &Self) -> bool {
        self.keys() == other.keys()
    }
}

impl std::hash::Hash for Diag {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.keys().hash(state);
    }
}

impl Diag {
    /// Creates a new `Diag` with a single message.
    #[track_caller]
    pub fn new<M: Into<DiagMsg>>(level: Level, msg: M) -> Self {
        Self::new_with_messages(level, vec![(msg.into(), Style::NoStyle)])
    }

    /// Creates a new `Diag` with multiple messages.
    #[track_caller]
    pub fn new_with_messages(level: Level, messages: Vec<(DiagMsg, Style)>) -> Self {
        Self {
            level,
            messages,
            code: None,
            span: MultiSpan::new(),
            children: vec![],
            // suggestions: Ok(vec![]),
            // args: Default::default(),
            // sort_span: DUMMY_SP,
            // is_lint: false,
            created_at: Location::caller(),
        }
    }

    /// Returns `true` if this diagnostic is an error.
    #[inline]
    pub fn is_error(&self) -> bool {
        self.level.is_error()
    }

    /// Formats the diagnostic messages into a single string.
    pub fn label(&self) -> Cow<'_, str> {
        flatten_messages(&self.messages, false, self.level)
    }

    /// Formats the diagnostic messages into a single string with ANSI color codes if applicable.
    pub fn label_with_style(&self, supports_color: bool) -> Cow<'_, str> {
        flatten_messages(&self.messages, supports_color, self.level)
    }

    /// Returns the messages of this diagnostic.
    pub fn messages(&self) -> &[(DiagMsg, Style)] {
        &self.messages
    }

    /// Returns the level of this diagnostic.
    pub fn level(&self) -> Level {
        self.level
    }

    /// Returns the code of this diagnostic as a string.
    pub fn id(&self) -> Option<String> {
        self.code.as_ref().map(|code| code.as_string())
    }

    /// Fields used for `PartialEq` and `Hash` implementations.
    fn keys(&self) -> impl PartialEq + std::hash::Hash + '_ {
        (
            &self.level,
            &self.messages,
            // self.args().collect(),
            &self.code,
            &self.span,
            // &self.suggestions,
            // (if self.is_lint { None } else { Some(&self.children) }),
            &self.children,
        )
    }
}

/// Setters.
impl Diag {
    /// Sets the span of this diagnostic.
    pub fn span(&mut self, span: impl Into<MultiSpan>) -> &mut Self {
        self.span = span.into();
        self
    }

    /// Sets the code of this diagnostic.
    pub fn code(&mut self, code: impl Into<DiagId>) -> &mut Self {
        self.code = Some(code.into());
        self
    }

    /// Adds a span/label to be included in the resulting snippet.
    ///
    /// This is pushed onto the [`MultiSpan`] that was created when the diagnostic
    /// was first built. That means it will be shown together with the original
    /// span/label, *not* a span added by one of the `span_{note,warn,help,suggestions}` methods.
    ///
    /// This span is *not* considered a ["primary span"][`MultiSpan`]; only
    /// the `Span` supplied when creating the diagnostic is primary.
    pub fn span_label(&mut self, span: Span, label: impl Into<DiagMsg>) -> &mut Self {
        self.span.push_span_label(span, label);
        self
    }

    /// Labels all the given spans with the provided label.
    /// See [`Self::span_label()`] for more information.
    pub fn span_labels(
        &mut self,
        spans: impl IntoIterator<Item = Span>,
        label: impl Into<DiagMsg>,
    ) -> &mut Self {
        let label = label.into();
        for span in spans {
            self.span_label(span, label.clone());
        }
        self
    }

    /// Adds a note with the location where this diagnostic was created and emitted.
    pub(crate) fn locations_note(&mut self, emitted_at: &Location<'_>) -> &mut Self {
        let msg = format!(
            "created at {},\n\
             emitted at {}",
            self.created_at, emitted_at
        );
        self.note(msg)
    }
}

/// Sub-diagnostics.
impl Diag {
    /// Add a warning attached to this diagnostic.
    pub fn warn(&mut self, msg: impl Into<DiagMsg>) -> &mut Self {
        self.sub(Level::Warning, msg, MultiSpan::new())
    }

    /// Prints the span with a warning above it.
    /// This is like [`Diag::warn()`], but it gets its own span.
    pub fn span_warn(&mut self, span: impl Into<MultiSpan>, msg: impl Into<DiagMsg>) -> &mut Self {
        self.sub(Level::Warning, msg, span)
    }

    /// Add a note to this diagnostic.
    pub fn note(&mut self, msg: impl Into<DiagMsg>) -> &mut Self {
        self.sub(Level::Note, msg, MultiSpan::new())
    }

    /// Prints the span with a note above it.
    /// This is like [`Diag::note()`], but it gets its own span.
    pub fn span_note(&mut self, span: impl Into<MultiSpan>, msg: impl Into<DiagMsg>) -> &mut Self {
        self.sub(Level::Note, msg, span)
    }

    pub fn highlighted_note(&mut self, messages: Vec<(impl Into<DiagMsg>, Style)>) -> &mut Self {
        self.sub_with_highlights(Level::Note, messages, MultiSpan::new())
    }

    /// Prints the span with a note above it.
    /// This is like [`Diag::note()`], but it gets emitted only once.
    pub fn note_once(&mut self, msg: impl Into<DiagMsg>) -> &mut Self {
        self.sub(Level::OnceNote, msg, MultiSpan::new())
    }

    /// Prints the span with a note above it.
    /// This is like [`Diag::note_once()`], but it gets its own span.
    pub fn span_note_once(
        &mut self,
        span: impl Into<MultiSpan>,
        msg: impl Into<DiagMsg>,
    ) -> &mut Self {
        self.sub(Level::OnceNote, msg, span)
    }

    /// Add a help message attached to this diagnostic.
    pub fn help(&mut self, msg: impl Into<DiagMsg>) -> &mut Self {
        self.sub(Level::Help, msg, MultiSpan::new())
    }

    /// Prints the span with a help above it.
    /// This is like [`Diag::help()`], but it gets its own span.
    pub fn help_once(&mut self, msg: impl Into<DiagMsg>) -> &mut Self {
        self.sub(Level::OnceHelp, msg, MultiSpan::new())
    }

    /// Add a help message attached to this diagnostic with a customizable highlighted message.
    pub fn highlighted_help(&mut self, msgs: Vec<(impl Into<DiagMsg>, Style)>) -> &mut Self {
        self.sub_with_highlights(Level::Help, msgs, MultiSpan::new())
    }

    /// Prints the span with some help above it.
    /// This is like [`Diag::help()`], but it gets its own span.
    pub fn span_help(&mut self, span: impl Into<MultiSpan>, msg: impl Into<DiagMsg>) -> &mut Self {
        self.sub(Level::Help, msg, span)
    }

    fn sub(
        &mut self,
        level: Level,
        msg: impl Into<DiagMsg>,
        span: impl Into<MultiSpan>,
    ) -> &mut Self {
        self.children.push(SubDiagnostic {
            level,
            messages: vec![(msg.into(), Style::NoStyle)],
            span: span.into(),
        });
        self
    }

    fn sub_with_highlights(
        &mut self,
        level: Level,
        messages: Vec<(impl Into<DiagMsg>, Style)>,
        span: MultiSpan,
    ) -> &mut Self {
        let messages = messages.into_iter().map(|(m, s)| (m.into(), s)).collect();
        self.children.push(SubDiagnostic { level, messages, span });
        self
    }
}

/// Flattens diagnostic messages, applying ANSI styles if requested.
fn flatten_messages(messages: &[(DiagMsg, Style)], with_style: bool, level: Level) -> Cow<'_, str> {
    if with_style {
        match messages {
            [] => Cow::Borrowed(""),
            [(msg, Style::NoStyle)] => Cow::Borrowed(msg.as_str()),
            [(msg, style)] => {
                let mut res = String::new();
                write_fmt(&mut res, msg, style, level);
                Cow::Owned(res)
            }
            messages => {
                let mut res = String::new();
                for (msg, style) in messages {
                    match style {
                        Style::NoStyle => res.push_str(msg.as_str()),
                        _ => write_fmt(&mut res, msg, style, level),
                    }
                }
                Cow::Owned(res)
            }
        }
    } else {
        match messages {
            [] => Cow::Borrowed(""),
            [(message, _)] => Cow::Borrowed(message.as_str()),
            messages => messages.iter().map(|(msg, _)| msg.as_str()).collect(),
        }
    }
}

fn write_fmt(output: &mut String, msg: &DiagMsg, style: &Style, level: Level) {
    let ansi_style = style.to_color_spec(level);
    write!(output, "{}{}{}", ansi_style.render(), msg.as_str(), ansi_style.render_reset()).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_styled_messages() {
        // Create a diagnostic with styled messages
        let mut diag = Diag::new(Level::Note, "test");

        diag.highlighted_note(vec![
            ("plain text ", Style::NoStyle),
            ("removed", Style::Removal),
            (" middle ", Style::NoStyle),
            ("added", Style::Addition),
        ]);

        let sub = &diag.children[0];

        // Without styles - just concatenated text
        let plain = sub.label();
        assert_eq!(plain, "plain text removed middle added");

        // With styles - includes ANSI escape codes
        let styled = sub.label_with_style(true);
        assert_eq!(
            styled.to_string(),
            "plain text \u{1b}[91mremoved\u{1b}[0m middle \u{1b}[92madded\u{1b}[0m".to_string()
        );
    }
}
