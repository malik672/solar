use crate::{BytePos, SessionGlobals};
use std::{
    cmp, fmt,
    ops::{Deref, DerefMut, Range},
};

/// A source code location.
///
/// Essentially a `lo..hi` range into a `SourceMap` file's source code.
///
/// Note that `lo` and `hi` are both offset from the file's starting position in the source map,
/// meaning that they are not always directly usable to index into the source string.
///
/// This is the case when there are multiple source files in the source map.
/// Use [`SourceMap::span_to_snippet`](crate::SourceMap::span_to_snippet) to get the actual source
/// code snippet of the span, or [`SourceMap::span_to_source`](crate::SourceMap::span_to_source) to
/// get the source file and source code range.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    lo: BytePos,
    hi: BytePos,
}

impl Default for Span {
    #[inline(always)]
    fn default() -> Self {
        Self::DUMMY
    }
}

impl Default for &Span {
    #[inline(always)]
    fn default() -> Self {
        &Span::DUMMY
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use the global `SourceMap` to print the span. If that's not
        // available, fall back to printing the raw values.

        fn fallback(span: Span, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Span({lo}..{hi})", lo = span.lo().0, hi = span.hi().0)
        }

        if SessionGlobals::is_set() {
            SessionGlobals::with(|g: &SessionGlobals| {
                let sm = g.source_map.lock();
                if let Some(source_map) = &*sm {
                    f.write_str(&source_map.span_to_diagnostic_string(*self))
                } else {
                    drop(sm);
                    fallback(*self, f)
                }
            })
        } else {
            fallback(*self, f)
        }
    }
}

impl Span {
    /// A dummy span.
    pub const DUMMY: Self = Self { lo: BytePos(0), hi: BytePos(0) };

    /// Creates a new span from two byte positions.
    #[inline]
    pub fn new(mut lo: BytePos, mut hi: BytePos) -> Self {
        if lo > hi {
            std::mem::swap(&mut lo, &mut hi);
        }
        Self { lo, hi }
    }

    /// Returns the span as a `Range<usize>`.
    ///
    /// Note that this may not be directly usable to index into the source string.
    /// See the [type-level documentation][Span] for more information.
    #[inline]
    pub fn to_range(self) -> Range<usize> {
        self.lo().to_usize()..self.hi().to_usize()
    }

    /// Returns the span as a `Range<u32>`.
    ///
    /// Note that this may not be directly usable to index into the source string.
    /// See the [type-level documentation][Span] for more information.
    #[inline]
    pub fn to_u32_range(self) -> Range<u32> {
        self.lo().to_u32()..self.hi().to_u32()
    }

    /// Returns the span's start position.
    ///
    /// Note that this may not be directly usable to index into the source string.
    /// See the [type-level documentation][Span] for more information.
    #[inline(always)]
    pub fn lo(self) -> BytePos {
        self.lo
    }

    /// Creates a new span with the same hi position as this span and the given lo position.
    #[inline]
    pub fn with_lo(self, lo: BytePos) -> Self {
        Self::new(lo, self.hi())
    }

    /// Returns the span's end position.
    ///
    /// Note that this may not be directly usable to index into the source string.
    /// See the [type-level documentation][Span] for more information.
    #[inline(always)]
    pub fn hi(self) -> BytePos {
        self.hi
    }

    /// Creates a new span with the same lo position as this span and the given hi position.
    #[inline]
    pub fn with_hi(self, hi: BytePos) -> Self {
        Self::new(self.lo(), hi)
    }

    /// Creates a new span representing an empty span at the beginning of this span.
    #[inline]
    pub fn shrink_to_lo(self) -> Self {
        Self::new(self.lo(), self.lo())
    }

    /// Creates a new span representing an empty span at the end of this span.
    #[inline]
    pub fn shrink_to_hi(self) -> Self {
        Self::new(self.hi(), self.hi())
    }

    /// Returns `true` if this is a dummy span.
    #[inline]
    pub fn is_dummy(self) -> bool {
        self == Self::DUMMY
    }

    /// Returns `true` if `self` fully encloses `other`.
    #[inline]
    pub fn contains(self, other: Self) -> bool {
        self.lo() <= other.lo() && other.hi() <= self.hi()
    }

    /// Returns `true` if `self` touches `other`.
    #[inline]
    pub fn overlaps(self, other: Self) -> bool {
        self.lo() < other.hi() && other.lo() < self.hi()
    }

    /// Returns `true` if `self` and `other` are equal.
    #[inline]
    pub fn is_empty(self, other: Self) -> bool {
        self.lo() == other.lo() && self.hi() == other.hi()
    }

    /// Splits a span into two composite spans around a certain position.
    #[inline]
    pub fn split_at(self, pos: u32) -> (Self, Self) {
        let len = self.hi().0 - self.lo().0;
        debug_assert!(pos <= len);

        let split_pos = BytePos(self.lo().0 + pos);
        (Self::new(self.lo(), split_pos), Self::new(split_pos, self.hi()))
    }

    /// Returns a `Span` that would enclose both `self` and `end`.
    ///
    /// Note that this can also be used to extend the span "backwards":
    /// `start.to(end)` and `end.to(start)` return the same `Span`.
    ///
    /// ```text
    ///     ____             ___
    ///     self lorem ipsum end
    ///     ^^^^^^^^^^^^^^^^^^^^
    /// ```
    #[inline]
    pub fn to(self, end: Self) -> Self {
        Self::new(cmp::min(self.lo(), end.lo()), cmp::max(self.hi(), end.hi()))
    }

    /// Returns a `Span` between the end of `self` to the beginning of `end`.
    ///
    /// ```text
    ///     ____             ___
    ///     self lorem ipsum end
    ///         ^^^^^^^^^^^^^
    /// ```
    #[inline]
    pub fn between(self, end: Self) -> Self {
        Self::new(self.hi(), end.lo())
    }

    /// Returns a `Span` from the beginning of `self` until the beginning of `end`.
    ///
    /// ```text
    ///     ____             ___
    ///     self lorem ipsum end
    ///     ^^^^^^^^^^^^^^^^^
    /// ```
    #[inline]
    pub fn until(self, end: Self) -> Self {
        Self::new(self.lo(), end.lo())
    }

    /// Joins all the spans in the given iterator using [`to`](Self::to).
    ///
    /// Returns [`DUMMY`](Self::DUMMY) if the iterator is empty.
    #[inline]
    pub fn join_many(spans: impl IntoIterator<Item = Self>) -> Self {
        spans.into_iter().reduce(Self::to).unwrap_or_default()
    }

    /// Joins the first and last span in the given iterator.
    ///
    /// Returns [`DUMMY`](Self::DUMMY) if the iterator is empty.
    #[inline]
    pub fn join_first_last(
        spans: impl IntoIterator<Item = Self, IntoIter: DoubleEndedIterator>,
    ) -> Self {
        let mut spans = spans.into_iter();
        let first = spans.next().unwrap_or_default();
        if let Some(last) = spans.next_back() { first.to(last) } else { first }
    }
}

/// A value paired with a source code location.
///
/// Wraps any value with a [`Span`] to track its location in the source code.
/// Implements `Deref` and `DerefMut` for transparent access to the inner value.
#[derive(Clone, Copy, Debug, Default)]
pub struct Spanned<T> {
    pub span: Span,
    pub data: T,
}

impl<T> Deref for Spanned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for Spanned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T> Spanned<T> {
    pub fn map<U, F>(self, f: F) -> Spanned<U>
    where
        F: FnOnce(T) -> U,
    {
        Spanned { span: self.span, data: f(self.data) }
    }

    pub fn as_ref(&self) -> Spanned<&T> {
        Spanned { span: self.span, data: &self.data }
    }

    pub fn as_mut(&mut self) -> Spanned<&mut T> {
        Spanned { span: self.span, data: &mut self.data }
    }

    pub fn into_inner(self) -> T {
        self.data
    }
}
