use std::cmp;
use std::fmt;
use std::marker;
use std::ops;

/// Represents the dimension of one axis of multi-dimensional data.
/// These types may represent a constant known at compile-time (e.g. `Const<3>`)
/// or runtime (e.g. `usize`),
pub trait Dim: 'static + Sized + Clone + Copy + fmt::Debug + Into<usize> + PartialEq {}

/// Represents a dimension known at compile time.
/// The primitive type `usize` is used for a dimension not known at compile time.
#[derive(Default, Clone, Copy)]
pub struct Const<const N: usize>;

impl<const N: usize> fmt::Debug for Const<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_const", N)
    }
}

impl<const N: usize> Dim for Const<N> {}

impl<const N: usize> From<Const<N>> for usize {
    fn from(_other: Const<N>) -> usize {
        N
    }
}

impl Dim for usize {}

impl<const N: usize> cmp::PartialEq<Const<N>> for usize {
    fn eq(&self, _other: &Const<N>) -> bool {
        *self == N
    }
}

impl<const N: usize> cmp::PartialEq<usize> for Const<N> {
    fn eq(&self, other: &usize) -> bool {
        N == *other
    }
}

impl<const N: usize> cmp::PartialEq<Const<N>> for Const<N> {
    fn eq(&self, _other: &Const<N>) -> bool {
        true
    }
}

pub trait AsIndex {
    type Index: Index;

    /// Get the value of this index in memory.
    /// For example, `(Const::<3>, 4).d()` is `[3, 4]`.
    ///
    /// One case where this is useful if you want to mutate the values--
    /// you can't increment the first dimension of `(Const::<3>, 4)`
    /// but you can increment the first dimension of `[3, 4]`.
    fn as_index(&self) -> Self::Index;
}

/// Represents dimensions of multi-dimensional data.
/// These types may represent constants known at compile-time (e.g. (Const<3>, Const<4>))
/// or runtime (e.g. [usize; 2])
/// or a mix: (usize, Const<2>)
///
/// Note that "index" and "shape" are conceptually the same type,
/// and this trait represents either.
pub trait Shape: 'static + Sized + Clone + Copy + fmt::Debug + AsIndex + ShapeEqBase<Self> {
    // Provided methods

    /// How many total elements are contained within a multi-dimensional data
    /// of this shape.
    /// For example, `[3, 4].num_elements()` is `12`
    fn num_elements(&self) -> usize {
        self.as_index().into_iter().product()
    }

    /// The stride values for multi-dimensional data of this shape.
    /// This is assuming the data is stored in a C-contiguous fashion,
    /// where the first axis changes slowest as you traverse the data
    /// (i.e. has the largest stride.)
    ///
    /// For example, [3, 4].strides() is `[4, 1]`
    /// since moving one unit along the first axis requires a stride of 4 elements,
    /// and moving one unit along the second axis requires a stride of 1 element.
    fn default_strides(&self) -> Self::Index {
        let mut result = Self::Index::zero();
        let mut acc = 1;
        for (r, d) in result
            .iter_mut()
            .rev()
            .zip(self.as_index().into_iter().rev())
        {
            *r = acc;
            acc *= d;
        }
        result
    }

    /// Panics if the given index is out-of-bounds for data of this shape.
    /// This should only be used for comparing shapes
    /// since the panic message uses that terminology
    fn out_of_bounds_fail(&self, idx: &Self::Index) {
        for (i, s) in idx.into_iter().zip(self.as_index().into_iter()) {
            if i >= s {
                panic!("Index out of bounds: index={idx:?} shape={self:?}");
            }
        }
    }

    /// Panics if the given index does not equal the given index.
    /// This should probably only be used for comparing shapes
    /// since the panic message uses that terminology
    fn shape_mismatch_fail<S: Shape>(&self, other: &S)
    where
        Self: ShapeEq<S>,
    {
        if !self.shape_eq(other) {
            panic!("Shapes do not match: {self:?} != {other:?}");
        }
    }
}

/// A sub-trait of Index that contains no compile-time constants (all components are stored in memory as `usize`.)
/// This opens up additional possibilities like indexing with brackets and `iter_mut()`.
pub trait Index:
    'static
    + Sized
    + Clone
    + Copy
    + fmt::Debug
    + ops::IndexMut<usize>
    + IntoIterator<Item = usize, IntoIter: DoubleEndedIterator + ExactSizeIterator>
    + AsIndex<Index = Self>
    + FromIndex<Self>
{
    // Required methods

    // TODO I don't think this method is well-formed for dynamic number of dimensions
    fn zero() -> Self;

    /// Converts this multi-dimensional index to a 1-dimensional index (usize.)
    /// No bounds checking is performed.
    fn to_i(&self, strides: &Self) -> usize {
        strides
            .into_iter()
            .zip(self.into_iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Returns an iterator over the components of this index,
    /// that allows modifying the components.
    fn iter_mut<'a>(&'a mut self) -> std::slice::IterMut<'a, usize>;
}

/// This index can be converted from another index.
/// This trait is implemented for all conversions that have some chance of success,
/// e.g. usize -> Const<3> (at runtime, the usize is checked to be 3.)
/// It is not implemented for conversions that have no chance of success at compile time
/// e.g. Const<4> -> Const<3>
pub trait ShapeEqBase<O>: Sized {
    // Required methods

    fn shape_eq(&self, other: &O) -> bool;
}

pub trait ShapeEq<T: Shape>: Shape<Index: IntoIndex<T::Index>> + ShapeEqBase<T> {}
impl<T: Shape<Index: IntoIndex<U::Index>> + ShapeEqBase<U>, U: Shape> ShapeEq<U> for T where
    U: ShapeEqBase<T>
{
}

/// This index can be converted from another index.
/// This trait is implemented for all conversions that have some chance of success,
/// e.g. usize -> Const<3> (at runtime, the usize is checked to be 3.)
/// It is not implemented for conversions that have no chance of success at compile time
/// e.g. Const<4> -> Const<3>
pub trait FromIndex<O: Index>: Sized {
    // Required methods

    /// Convert to this index from the given compile-time compatible index,
    /// returning None if they turn out not to be compatible at runtime.
    fn try_from_index(other: O) -> Option<Self>;

    // Provided methods

    /// Convert to this index from the given compile-time compatible index,
    /// panicking if they turn out not to be compatible at runtime.
    fn from_index_fail(other: O) -> Self {
        Self::try_from_index(other).unwrap_or_else(|| {
            panic!(
                "Indices are not compatible: type={} got={other:?}",
                std::any::type_name::<Self>()
            )
        })
    }
}

pub trait IntoIndex<T>: Index {
    // Required methods

    /// Convert from this index into the given compile-time compatible index,
    /// returning None if they turn out not to be compatible at runtime.
    fn try_into_index(self) -> Option<T>;

    // Provided methods

    /// Convert from this index into the given compile-time compatible index,
    /// panicking if they turn out not to be compatible at runtime.
    fn into_index_fail(self) -> T;
}

impl<T: Index, U> IntoIndex<U> for T
where
    U: FromIndex<T>,
{
    // Required methods

    fn try_into_index(self) -> Option<U> {
        U::try_from_index(self)
    }

    // Provided methods

    fn into_index_fail(self) -> U {
        U::from_index_fail(self)
    }
}

impl<const N: usize> AsIndex for [usize; N] {
    type Index = [usize; N];

    fn as_index(&self) -> Self::Index {
        *self
    }
}

impl<const N: usize> Shape for [usize; N] {}

impl<const N: usize> ShapeEqBase<[usize; N]> for [usize; N] {
    fn shape_eq(&self, other: &[usize; N]) -> bool {
        self == other
    }
}

impl<const N: usize> Index for [usize; N] {
    fn zero() -> Self {
        [0; N]
    }

    fn iter_mut<'a>(&'a mut self) -> std::slice::IterMut<'a, usize> {
        <[usize]>::iter_mut(self)
    }
}

impl<const N: usize> FromIndex<[usize; N]> for [usize; N] {
    fn try_from_index(other: [usize; N]) -> Option<[usize; N]> {
        Some(other)
    }
}

impl<D1: Dim> AsIndex for (D1,) {
    type Index = [usize; 1];

    fn as_index(&self) -> Self::Index {
        [self.0.into()]
    }
}

impl<D1: Dim> Shape for (D1,) {}

impl<D1: Dim + cmp::PartialEq<E1>, E1: Dim> ShapeEqBase<(E1,)> for (D1,) {
    fn shape_eq(&self, other: &(E1,)) -> bool {
        self.0 == other.0
    }
}

impl<D1: Dim + cmp::PartialEq<usize>> ShapeEqBase<[usize; 1]> for (D1,) {
    fn shape_eq(&self, other: &[usize; 1]) -> bool {
        self.0 == other[0]
    }
}

impl<D1: Dim> ShapeEqBase<(D1,)> for [usize; 1]
where
    usize: PartialEq<D1>,
{
    fn shape_eq(&self, other: &(D1,)) -> bool {
        self[0] == other.0
    }
}

impl<D1: Dim, D2: Dim> AsIndex for (D1, D2) {
    type Index = [usize; 2];

    fn as_index(&self) -> Self::Index {
        [self.0.into(), self.1.into()]
    }
}

impl<D1: Dim, D2: Dim> Shape for (D1, D2) {}

impl<D1: Dim + PartialEq<E1>, D2: Dim + PartialEq<E2>, E1: Dim, E2: Dim> ShapeEqBase<(E1, E2)>
    for (D1, D2)
{
    fn shape_eq(&self, other: &(E1, E2)) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<D1: Dim + PartialEq<usize>, D2: Dim + PartialEq<usize>> ShapeEqBase<[usize; 2]> for (D1, D2) {
    fn shape_eq(&self, other: &[usize; 2]) -> bool {
        self.0 == other[0] && self.1 == other[1]
    }
}

impl<D1: Dim, D2: Dim> ShapeEqBase<(D1, D2)> for [usize; 2]
where
    usize: PartialEq<D1>,
    usize: PartialEq<D2>,
{
    fn shape_eq(&self, other: &(D1, D2)) -> bool {
        self[0] == other.0 && self[1] == other.1
    }
}

/////////////////////////////////////////////

pub trait DefiniteRange: Sized {
    type Item: Index;

    fn first(&self) -> Option<Self::Item>;
    fn next(&self, cur: Self::Item) -> Option<Self::Item>;

    fn nd_iter(self) -> RangeIter<Self> {
        let cur = self.first();
        RangeIter { range: self, cur }
    }
}

impl<I: AsIndex> DefiniteRange for ops::RangeTo<I> {
    type Item = I::Index;

    fn first(&self) -> Option<Self::Item> {
        self.end
            .as_index()
            .into_iter()
            .all(|n| n > 0)
            .then_some(I::Index::zero())
    }

    fn next(&self, mut cur: Self::Item) -> Option<Self::Item> {
        for (i, n) in cur
            .iter_mut()
            .rev()
            .zip(self.end.as_index().into_iter().rev())
        {
            *i += 1;
            if *i < n {
                return Some(cur);
            } else {
                // Iteration finished in this axis
                *i = 0;
            }
        }
        None
    }
}

impl<I: AsIndex> DefiniteRange for ops::RangeToInclusive<I> {
    type Item = I::Index;

    fn first(&self) -> Option<Self::Item> {
        Some(I::Index::zero())
    }

    fn next(&self, mut cur: Self::Item) -> Option<Self::Item> {
        for (i, n) in cur
            .iter_mut()
            .rev()
            .zip(self.end.as_index().into_iter().rev())
        {
            *i += 1;
            if *i <= n {
                return Some(cur);
            } else {
                // Iteration finished in this axis
                *i = 0;
            }
        }
        None
    }
}

impl<I: AsIndex> DefiniteRange for ops::Range<I> {
    type Item = I::Index;

    fn first(&self) -> Option<Self::Item> {
        self.start
            .as_index()
            .into_iter()
            .zip(self.end.as_index().into_iter())
            .all(|(s, e)| e > s)
            .then_some(self.start.as_index())
    }

    fn next(&self, mut cur: Self::Item) -> Option<Self::Item> {
        for (i, (s, e)) in cur.iter_mut().rev().zip(
            self.start
                .as_index()
                .into_iter()
                .rev()
                .zip(self.end.as_index().into_iter().rev()),
        ) {
            *i += 1;
            if *i < e {
                return Some(cur);
            } else {
                // Iteration finished in this axis
                *i = s;
            }
        }
        None
    }
}

impl<I: Shape> DefiniteRange for ops::RangeInclusive<I> {
    type Item = I::Index;

    fn first(&self) -> Option<Self::Item> {
        self.start()
            .as_index()
            .into_iter()
            .zip(self.end().as_index().into_iter())
            .all(|(s, e)| e >= s)
            .then_some(self.start().as_index())
    }

    fn next(&self, mut cur: Self::Item) -> Option<Self::Item> {
        for (i, (s, e)) in cur.iter_mut().rev().zip(
            self.start()
                .as_index()
                .into_iter()
                .rev()
                .zip(self.end().as_index().into_iter().rev()),
        ) {
            *i += 1;
            if *i <= e {
                return Some(cur);
            } else {
                // Iteration finished in this axis
                *i = s;
            }
        }
        None
    }
}

/////////////////////////////////////////////

// Wrappers for array data
// Three kinds: owned, ref, mut

/// A multi-dimensional array.
///
/// This type owns the underlying data.
#[derive(Debug, Clone)]
pub struct Array<S: Shape, E, D> {
    shape: S,
    element: marker::PhantomData<E>,
    offset: usize,
    strides: S::Index,
    data: D,
}

/// A view into multi-dimensional data.
///
/// This type holds an immutable reference the underlying data.
/// For a mutable reference, see [ViewMut],
/// or for owned data, see [Array].
#[derive(Debug)]
pub struct View<'a, S: Shape, E, D: ?Sized> {
    shape: S,
    element: marker::PhantomData<E>,
    offset: usize,
    strides: S::Index,
    data: &'a D,
}

/// A mutable view into multi-dimensional data.
///
/// This type holds an mutable reference the underlying data.
/// For an immutable reference, see [View],
/// or for owned data, see [Array].
#[derive(Debug)]
pub struct ViewMut<'a, S: Shape, E, D: ?Sized> {
    shape: S,
    element: marker::PhantomData<E>,
    offset: usize,
    strides: S::Index,
    data: &'a mut D,
}

/// This trait marks anything which can represent a read-only view into multi-dimensional data,
/// such as:
/// * Array
/// * View
/// * ViewMut
///
/// Using this trait allows functions to accept any kind of read-only multi-dimensional data. For example:
/// ```
/// fn sum(a: &impl IntoView<Element=f32, Data: AsRef<[f32]>>) -> f32 {
///     let a = a.view(); // Convert to concrete type View
///
///     a.into_iter().sum()
/// }
/// ```
///
/// This `sum` function can now accept `&Array`, `&View`, or `&ViewMut`.

pub trait IntoView<S: Shape, E, D: ?Sized> {
    fn view_shape(&self, shape: S) -> View<S, E, D>;
}

pub trait IntoSizedView<E, D: ?Sized>: IntoView<Self::NativeShape, E, D> {
    type NativeShape: Shape;

    fn view(&self) -> View<Self::NativeShape, E, D>;
}

impl<S: Shape, S2: Shape + ShapeEq<S>, E, D: ?Sized, D2: AsRef<D>> IntoView<S, E, D>
    for Array<S2, E, D2>
{
    fn view_shape(&self, shape: S) -> View<S, E, D> {
        self.shape.shape_mismatch_fail(&shape);
        View {
            shape,
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.into_index_fail(),
            data: self.data.as_ref(),
        }
    }
}

impl<S: Shape, E, D: ?Sized, D2: AsRef<D>> IntoSizedView<E, D> for Array<S, E, D2> {
    type NativeShape = S;

    fn view(&self) -> View<S, E, D> {
        View {
            shape: self.shape.clone(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.clone(),
            data: self.data.as_ref(),
        }
    }
}

impl<S: Shape, S2: Shape + ShapeEq<S>, E, D: ?Sized, D2: ?Sized + AsRef<D>> IntoView<S, E, D>
    for View<'_, S2, E, D2>
{
    fn view_shape(&self, shape: S) -> View<S, E, D> {
        self.shape.shape_mismatch_fail(&shape);
        View {
            shape,
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.into_index_fail(),
            data: self.data.as_ref(),
        }
    }
}

impl<S: Shape, E, D: ?Sized, D2: ?Sized + AsRef<D>> IntoSizedView<E, D> for View<'_, S, E, D2> {
    type NativeShape = S;

    fn view(&self) -> View<S, E, D> {
        View {
            shape: self.shape.clone(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.clone(),
            data: self.data.as_ref(),
        }
    }
}

impl<S: Shape, S2: Shape + ShapeEq<S>, E, D: ?Sized, D2: ?Sized + AsRef<D>> IntoView<S, E, D>
    for ViewMut<'_, S2, E, D2>
{
    fn view_shape(&self, shape: S) -> View<S, E, D> {
        self.shape.shape_mismatch_fail(&shape);
        View {
            shape,
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.into_index_fail(),
            data: self.data.as_ref(),
        }
    }
}

impl<S: Shape, E, D: ?Sized, D2: ?Sized + AsRef<D>> IntoSizedView<E, D> for ViewMut<'_, S, E, D2> {
    type NativeShape = S;

    fn view(&self) -> View<S, E, D> {
        View {
            shape: self.shape.clone(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.clone(),
            data: self.data.as_ref(),
        }
    }
}

/// This trait marks anything which can represent a mutable view into multi-dimensional data,
/// such as:
/// * Array
/// * ViewMut
///
/// Using this trait allows functions to accept any kind of mutable multi-dimensional data. For example:
/// ```
/// fn increment(a: &mut impl IntoView<Element=i32, Data: AsRef<[i32]> + AsMut<[i32]>>) -> i32 {
///     let mut a = a.view_mut(); // Convert to concrete type View
///
///     for e in a.into_iter() {
///         e += 1;
///     }
/// }
/// ```
///
/// This `increment` function can now accept `&mut Array` or `&mut View`.
pub trait IntoViewMut<S: Shape, E, D: ?Sized>: IntoView<S, E, D> {
    fn view_mut_shape(&mut self, shape: S) -> ViewMut<'_, S, E, D>;
}

pub trait IntoSizedViewMut<E, D: ?Sized>:
    IntoSizedView<E, D> + IntoViewMut<Self::NativeShape, E, D>
{
    fn view_mut(&mut self) -> ViewMut<Self::NativeShape, E, D>;
}

impl<S: Shape, S2: Shape + ShapeEq<S>, E, D: ?Sized, D2: AsRef<D> + AsMut<D>> IntoViewMut<S, E, D>
    for Array<S2, E, D2>
{
    fn view_mut_shape(&mut self, shape: S) -> ViewMut<'_, S, E, D> {
        self.shape.shape_mismatch_fail(&shape);
        ViewMut {
            shape,
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.into_index_fail(), // Shouldn't fail if shapes match
            data: self.data.as_mut(),
        }
    }
}

impl<S: Shape, E, D: ?Sized, D2: AsRef<D> + AsMut<D>> IntoSizedViewMut<E, D> for Array<S, E, D2> {
    fn view_mut(&mut self) -> ViewMut<'_, S, E, D> {
        ViewMut {
            shape: self.shape.clone(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.clone(),
            data: self.data.as_mut(),
        }
    }
}

impl<S: Shape, S2: Shape + ShapeEq<S>, E, D: ?Sized, D2: ?Sized + AsRef<D> + AsMut<D>>
    IntoViewMut<S, E, D> for ViewMut<'_, S2, E, D2>
{
    fn view_mut_shape(&mut self, shape: S) -> ViewMut<'_, S, E, D> {
        self.shape.shape_mismatch_fail(&shape);
        ViewMut {
            shape,
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.into_index_fail(), // Shouldn't fail if shapes match
            data: self.data.as_mut(),
        }
    }
}

impl<S: Shape, E, D: ?Sized, D2: ?Sized + AsRef<D> + AsMut<D>> IntoSizedViewMut<E, D>
    for ViewMut<'_, S, E, D2>
{
    fn view_mut(&mut self) -> ViewMut<'_, S, E, D> {
        ViewMut {
            shape: self.shape.clone(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.clone(),
            data: self.data.as_mut(),
        }
    }
}

/////////////////////////////////////////////

macro_rules! impl_view_methods {
    ($struct:ident $(<$lt:lifetime>)?) => {
        impl<S: Shape, E, D: AsRef<[E]>> $struct<$($lt,)? S, E, D> {
            pub unsafe fn get_unchecked(&self, idx: S::Index) -> &E {
                self.data.as_ref().get_unchecked(self.offset + idx.to_i(&self.strides))
            }
        }

        impl<S: Shape, E, D: AsRef<[E]>> ops::Index<S::Index> for $struct<$($lt,)? S, E, D> {
            type Output = E;

            fn index(&self, idx: S::Index) -> &E {
                self.shape.out_of_bounds_fail(&idx);
                unsafe { self.get_unchecked(idx) }
            }
        }

        impl<'a, S: Shape, E, D: AsRef<[E]>> IntoIterator for &'a $struct<$($lt,)? S, E, D> {
            type Item = &'a E;
            type IntoIter = NdIter<'a, S, E>;

            fn into_iter(self) -> Self::IntoIter {
                NdIter {
                    shape: self.shape.clone(),
                    strides: self.strides,
                    data: &self.data.as_ref()[self.offset..],
                    idx: (..self.shape).first(),
                }
            }
        }
    };
}

macro_rules! impl_view_mut_methods {
    ($struct:ident $(<$lt:lifetime>)?) => {
        impl<S: Shape, E, D: AsRef<[E]> + AsMut<[E]>> $struct<$($lt,)? S, E, D> {
            pub unsafe fn get_unchecked_mut(&mut self, idx: S::Index) -> &mut E {
                self.data.as_mut().get_unchecked_mut(self.offset + idx.to_i(&self.strides))
            }
        }

        impl<S: Shape, E, D: AsRef<[E]> + AsMut<[E]>> ops::IndexMut<S::Index> for $struct<$($lt,)? S, E, D> {
            fn index_mut(&mut self, idx: S::Index) -> &mut E {
                self.shape.out_of_bounds_fail(&idx);
                unsafe { self.get_unchecked_mut(idx) }
            }
        }

        impl<'a, S: Shape, E, D: AsRef<[E]> + AsMut<[E]>> IntoIterator for &'a mut $struct<$($lt,)? S, E, D> {
            type Item = &'a mut E;
            type IntoIter = NdIterMut<'a, S, E>;

            fn into_iter(self) -> Self::IntoIter {
                NdIterMut {
                    shape: self.shape.clone(),
                    strides: self.strides,
                    data: &mut self.data.as_mut()[self.offset..],
                    idx: (..self.shape).first(),
                }
            }
        }
    };
}

impl_view_methods!(View<'_>);
impl_view_methods!(ViewMut<'_>);
impl_view_methods!(Array);

impl_view_mut_methods!(ViewMut<'_>);
impl_view_mut_methods!(Array);

/////////////////////////////////////////////

pub struct NdIter<'a, S: Shape, E> {
    shape: S,
    strides: S::Index,
    data: &'a [E],
    idx: Option<S::Index>,
}

impl<'a, S: Shape, E> NdIter<'a, S, E> {
    pub fn nd_enumerate(self) -> NdEnumerate<Self> {
        NdEnumerate(self)
    }
}

impl<'a, S: Shape, E> Iterator for NdIter<'a, S, E> {
    type Item = &'a E;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx?;

        let val = unsafe { self.data.get_unchecked(idx.to_i(&self.strides)) };
        self.idx = (..self.shape).next(idx);
        Some(val)
    }
}

pub struct NdIterMut<'a, S: Shape, E> {
    shape: S,
    strides: S::Index,
    data: &'a mut [E],
    idx: Option<S::Index>,
}

impl<'a, S: Shape, E> NdIterMut<'a, S, E> {
    pub fn nd_enumerate(self) -> NdEnumerate<Self> {
        NdEnumerate(self)
    }
}

impl<'a, S: Shape, E> Iterator for NdIterMut<'a, S, E> {
    type Item = &'a mut E;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx?;
        let val = unsafe { &mut *(self.data.get_unchecked_mut(idx.to_i(&self.strides)) as *mut E) };
        self.idx = (..self.shape).next(idx);
        Some(val)
    }
}

pub struct NdEnumerate<I>(I);

impl<'a, S: Shape, E> Iterator for NdEnumerate<NdIter<'a, S, E>> {
    type Item = (S::Index, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.0.idx?;
        let val = unsafe { self.0.data.get_unchecked(idx.to_i(&self.0.strides)) };
        self.0.idx = (..self.0.shape).next(idx);
        Some((idx, val))
    }
}

impl<'a, S: Shape, E> Iterator for NdEnumerate<NdIterMut<'a, S, E>> {
    type Item = (S::Index, &'a mut E);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.0.idx?;
        let val =
            unsafe { &mut *(self.0.data.get_unchecked_mut(idx.to_i(&self.0.strides)) as *mut E) };
        self.0.idx = (..self.0.shape).next(idx);
        Some((idx, val))
    }
}

/////////////////////////////////////////////

/// This trait marks anything which can serve as an output target for multi-dimensional data,
/// such as:
/// * `&mut Array`
/// * `&mut ViewMut`
/// * `Alloc` (a marker struct indicating that the output should be allocated on the heap at
///   runtime.)
///
/// Using this trait allows functions to store their results in either an existing array
/// or a newly allocated array, at the discretion of the caller.
///
/// Here is an example:
/// ```
/// fn ones<S: Shape>(shape: S, out: impl IntoTarget<Element=i32, Data: AsRef<[i32]> + AsMut<[i32]>>) -> i32 {
///     let mut a = a.view_mut(); // Convert to concrete type View
///
///     for e in a.into_iter() {
///         e += 1;
///     }
/// }
/// ```
///
/// This `increment` function can now accept `&mut Array` or `&mut ViewMut`.
pub trait IntoTarget<S: Shape> {
    type Target: OutTarget;

    fn build_shape(self, shape: S) -> Self::Target;
}

pub trait IntoSizedTarget: IntoTarget<Self::NativeShape> {
    type NativeShape: Shape;

    fn build(self) -> Self::Target;
}

pub trait OutTarget {
    type Output;

    fn output(self) -> Self::Output;
}

// TODO why are the reflexive bounds needed?
impl<'a, S: Shape, S2: Shape + ShapeEq<S>, E, D: AsRef<D> + AsMut<D>> IntoTarget<S>
    for &'a mut Array<S2, E, D>
{
    type Target = ViewMut<'a, S, E, D>;

    fn build_shape(self, shape: S) -> Self::Target {
        self.view_mut_shape(shape)
    }
}

// TODO why are the reflexive bounds needed?
impl<'a, S: Shape, E, D: AsRef<D> + AsMut<D>> IntoSizedTarget for &'a mut Array<S, E, D> {
    type NativeShape = S;

    fn build(self) -> Self::Target {
        self.view_mut()
    }
}

// TODO why are the reflexive bounds needed?
impl<'a, S: Shape, S2: Shape + ShapeEq<S>, E, D: ?Sized + AsRef<D> + AsMut<D>> IntoTarget<S>
    for &'a mut ViewMut<'a, S2, E, D>
{
    type Target = ViewMut<'a, S, E, D>;

    fn build_shape(self, shape: S) -> Self::Target {
        self.view_mut_shape(shape)
    }
}

// TODO why are the reflexive bounds needed?
impl<'a, S: Shape, E, D: ?Sized + AsRef<D> + AsMut<D>> IntoSizedTarget
    for &'a mut ViewMut<'a, S, E, D>
{
    type NativeShape = S;

    fn build(self) -> Self::Target {
        self.view_mut()
    }
}

// TODO why are the reflexive bounds needed?
impl<'a, S: Shape, E, D: ?Sized + AsRef<D> + AsMut<D>> OutTarget for ViewMut<'a, S, E, D> {
    type Output = ();

    fn output(self) -> Self::Output {
        ()
    }
}

// TODO Get rid of phatom data!
pub struct AllocShape<S: Shape, E> {
    shape: S,
    element: marker::PhantomData<E>,
}

/// An output target wrapping an owned [Array],
/// that when viewed has a different (but equal) shape
/// than the underlying `Array`
pub struct ArrayTarget<S: Shape, S2: Shape + ShapeEq<S>, E, D> {
    array: Array<S2, E, D>,
    shape: S,
}

impl<S: Shape, S2: Shape + ShapeEq<S>, S3: Shape + ShapeEq<S2>, E, D: ?Sized, D2: AsRef<D>>
    IntoView<S, E, D> for ArrayTarget<S2, S3, E, D2>
{
    fn view_shape(&self, shape: S) -> View<S, E, D> {
        self.shape.shape_mismatch_fail(&shape);
        View {
            shape,
            element: marker::PhantomData,
            offset: self.array.offset,
            strides: self.array.strides.into_index_fail().into_index_fail(),
            data: self.array.data.as_ref(),
        }
    }
}

impl<S: Shape, S2: Shape + ShapeEq<S>, E, D: ?Sized, D2: AsRef<D>> IntoSizedView<E, D>
    for ArrayTarget<S, S2, E, D2>
{
    type NativeShape = S;

    fn view(&self) -> View<S, E, D> {
        View {
            shape: self.shape.clone(),
            element: marker::PhantomData,
            offset: self.array.offset,
            strides: self.array.strides.into_index_fail(),
            data: self.array.data.as_ref(),
        }
    }
}

impl<
        S: Shape,
        S2: Shape + ShapeEq<S>,
        S3: Shape + ShapeEq<S2>,
        E,
        D: ?Sized,
        D2: AsRef<D> + AsMut<D>,
    > IntoViewMut<S, E, D> for ArrayTarget<S2, S3, E, D2>
{
    fn view_mut_shape(&mut self, shape: S) -> ViewMut<'_, S, E, D> {
        self.shape.shape_mismatch_fail(&shape);
        ViewMut {
            shape,
            element: marker::PhantomData,
            offset: self.array.offset,
            strides: self.array.strides.into_index_fail().into_index_fail(),
            data: self.array.data.as_mut(),
        }
    }
}

impl<S: Shape, S2: Shape + ShapeEq<S>, E, D: ?Sized, D2: AsRef<D> + AsMut<D>> IntoSizedViewMut<E, D>
    for ArrayTarget<S, S2, E, D2>
{
    fn view_mut(&mut self) -> ViewMut<'_, S, E, D> {
        ViewMut {
            shape: self.shape.clone(),
            element: marker::PhantomData,
            offset: self.array.offset,
            strides: self.array.strides.into_index_fail(),
            data: self.array.data.as_mut(),
        }
    }
}

impl<'a, S: Shape, S2: Shape + ShapeEq<S>, E, D: AsRef<D> + AsMut<D>> OutTarget
    for ArrayTarget<S, S2, E, D>
{
    type Output = Array<S2, E, D>;

    fn output(self) -> Self::Output {
        self.array
    }
}

impl<'a, S: Shape, S2: Shape + ShapeEq<S>, E: Default + Clone> IntoTarget<S> for AllocShape<S2, E> {
    type Target = ArrayTarget<S, S2, E, Vec<E>>;

    fn build_shape(self, shape: S) -> Self::Target {
        let AllocShape {
            shape: self_shape, ..
        } = self;
        self_shape.shape_mismatch_fail(&shape);
        ArrayTarget {
            shape,
            array: Array {
                shape: self_shape,
                strides: self_shape.default_strides(),
                element: marker::PhantomData,
                offset: 0,
                data: vec![E::default(); self_shape.num_elements()],
            },
        }
    }
}

impl<'a, S: Shape, E: Default + Clone> IntoSizedTarget for AllocShape<S, E> {
    type NativeShape = S;

    fn build(self) -> Self::Target {
        let AllocShape { shape, .. } = self;
        ArrayTarget {
            shape,
            array: Array {
                shape,
                strides: shape.default_strides(),
                element: marker::PhantomData,
                offset: 0,
                data: vec![E::default(); shape.num_elements()],
            },
        }
    }
}

pub struct Alloc<E> {
    element: marker::PhantomData<E>,
}

impl<'a, S: Shape, E: Default + Clone> IntoTarget<S> for Alloc<E> {
    type Target = Array<S, E, Vec<E>>;

    fn build_shape(self, shape: S) -> Self::Target {
        Array {
            shape,
            strides: shape.default_strides(),
            element: marker::PhantomData,
            offset: 0,
            data: vec![E::default(); shape.num_elements()],
        }
    }
}

impl<S: Shape, E, D> OutTarget for Array<S, E, D> {
    type Output = Self;

    fn output(self) -> Self::Output {
        self
    }
}

/////////////////////////////////////////////

pub struct RangeIter<R: DefiniteRange> {
    range: R,
    cur: Option<R::Item>,
}

impl<R: DefiniteRange> Iterator for RangeIter<R> {
    type Item = R::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.cur?;
        self.cur = self.range.next(cur);
        Some(cur)
    }
}

/////////////////////////////////////////////

#[cfg(test)]
mod test {

    use crate::{
        Alloc, AllocShape, Array, AsIndex, Const, DefiniteRange, IntoSizedTarget, IntoSizedView,
        IntoSizedViewMut, IntoTarget, OutTarget, Shape, ShapeEqBase,
    };
    use std::marker::PhantomData;

    #[test]
    fn test_shapes() {
        let s = (Const::<3>, Const::<4>);
        assert!(s.shape_eq(&(Const::<3>, Const::<4>)));
        assert!(s.shape_eq(&(Const::<3>, 4)));
        assert!(s.shape_eq(&(3, 4)));
        assert!(s.shape_eq(&[3, 4]));
        assert!(!s.shape_eq(&(Const::<3>, 5)));
        assert!(!s.shape_eq(&(3, 5)));
        assert!(!s.shape_eq(&[3, 5]));

        assert!((Const::<3>, Const::<4>).shape_eq(&s));
        assert!((Const::<3>, 4).shape_eq(&s));
        assert!((3, 4).shape_eq(&s));
        assert!([3, 4].shape_eq(&s));
        assert!(!(Const::<3>, 5).shape_eq(&s));
        assert!(!(3, 5).shape_eq(&s));
        assert!(![3, 5].shape_eq(&s));
    }

    #[test]
    fn test() {
        let mut t = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
            element: PhantomData::<i32>,
            offset: 0,
            data: vec![1, 2, 3, 4],
        };

        for e in &t {
            dbg!(e);
        }
        for e in &mut t {
            *e *= 2;
        }
        for e in &t {
            dbg!(e);
        }
    }

    #[test]
    fn test_bernstein() {
        /// Compute the binomial coefficient (n choose k)
        fn binomial(n: usize, k: usize) -> usize {
            let numerator: usize = (1..=k).map(|i| n + 1 - i).product();
            let denominator: usize = (1..=k).product();
            numerator / denominator
        }

        fn bernstein_coef<
            V: IntoSizedView<f32, [f32]>,
            O: IntoTarget<V::NativeShape, Target: IntoSizedViewMut<f32, [f32]>>,
        >(
            c_m: &V,
            out: O,
        ) -> <O::Target as OutTarget>::Output {
            let c_m = c_m.view();

            let mut target = out.build_shape(c_m.shape);
            let mut out_view = target.view_mut();

            // XXX
            //let val1 = c_m[<<V as IntoView>::Shape as Shape>::Index::zero()];

            for (i, out_entry) in (&mut out_view).into_iter().nd_enumerate() {
                *out_entry = (..=i)
                    .nd_iter()
                    .map(|j| {
                        let num: usize = i
                            .into_iter()
                            .zip(j.into_iter())
                            .map(|(i_n, j_n)| binomial(i_n, j_n))
                            .product();
                        let den: usize = c_m
                            .shape
                            .as_index()
                            .into_iter()
                            .zip(j.into_iter())
                            .map(|(d_n, j_n)| binomial(d_n, j_n))
                            .product();
                        let b = (num as f32) / (den as f32);
                        b * c_m[j]
                    })
                    .sum();
            }

            target.output()
        }

        // TEST DATA

        let a = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
            element: PhantomData,
            offset: 0,
            data: vec![1., 2., 3., 4.],
        };

        /*let mut b = Array {
            shape: [2, 2],
            strides: [2, 2].default_strides(),
            element: PhantomData,
            offset: 0,
            data: vec![0.; 4],
        };*/

        //bernstein_coef(&a, &mut b.view_mut());
        //bernstein_coef(&a.view(), &mut b);

        dbg!(bernstein_coef(
            &a,
            Alloc {
                element: PhantomData
            }
        ));
        dbg!(bernstein_coef(
            &a,
            AllocShape {
                shape: [2, 2],
                element: PhantomData
            }
        ));
        panic!();

        //bernstein_coef(&a, &mut b);
    }

    #[test]
    fn test_sum() {
        fn sum(in1: &impl IntoSizedView<f32, [f32]>) -> f32 {
            let in1 = in1.view();

            in1.into_iter().sum()
        }

        let a = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
            element: PhantomData,
            offset: 0,
            data: vec![1., 2., 3., 4.],
        };

        sum(&a);
    }

    #[test]
    fn test_ones() {
        fn ones<O: IntoSizedTarget<Target: IntoSizedViewMut<f32, [f32]>>>(
            out: O,
        ) -> <O::Target as OutTarget>::Output {
            let mut target = out.build();
            let mut out_view = target.view_mut();

            for e in &mut out_view {
                *e = 1.;
            }

            target.output()
        }

        let mut a = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
            element: PhantomData::<f32>,
            offset: 0,
            data: vec![1., 2., 3., 4.],
        };

        ones(&mut a);
        ones(AllocShape {
            shape: [4, 4],
            element: PhantomData,
        });
    }
}
