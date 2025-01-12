use std::fmt;
use std::marker;
use std::ops;

/// Represents the dimension of one axis of multi-dimensional data.
/// These types may represent a constant known at compile-time (e.g. `Const<3>`)
/// or runtime (e.g. `usize`),
pub trait Dim:
    'static
    + Sized
    + Clone
    + Copy
    + fmt::Debug
    + PartialEq
    + Eq
    + Into<usize>
    + FromDim<Self>
    + FromDim<usize>
{
}

/// This dimension or index can be converted from another dimension or index.
/// This trait is implemented for all conversions that have some chance of success,
/// e.g. usize -> Const<3> (at runtime, the usize is checked to be 3.)
/// It is not implemented for conversions that have no chance of success at compile time
/// e.g. Const<4> -> Const<3>
pub trait FromDim<O: Dim>: Sized {
    fn try_from_dim(other: O) -> Option<Self>;
}

/// This dimension or index can be converted into another dimension or index.
/// See [FromDim]
pub trait IntoDim<T>: Dim {
    fn try_into_dim(self) -> Option<T>;
}

impl<T: Dim, U> IntoDim<U> for T
where
    U: FromDim<T>,
{
    fn try_into_dim(self) -> Option<U> {
        U::try_from_dim(self)
    }
}

/// Represents a dimension known at compile time.
/// The primitive type `usize` is used for a dimension not known at compile time.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct Const<const N: usize>;

impl<const N: usize> fmt::Debug for Const<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_const", N)
    }
}

impl<const N: usize> Dim for Const<N> {}

impl Dim for usize {}

impl<const N: usize> From<Const<N>> for usize {
    fn from(_: Const<N>) -> Self {
        N
    }
}

impl<const N: usize> FromDim<Const<N>> for usize {
    fn try_from_dim(_: Const<N>) -> Option<Self> {
        Some(N)
    }
}

impl<const N: usize> FromDim<usize> for Const<N> {
    fn try_from_dim(other: usize) -> Option<Self> {
        if other == N {
            Some(Const::<N>)
        } else {
            None
        }
    }
}

impl FromDim<usize> for usize {
    fn try_from_dim(other: usize) -> Option<usize> {
        Some(other)
    }
}

impl<const N: usize> FromDim<Const<N>> for Const<N> {
    fn try_from_dim(other: Const<N>) -> Option<Const<N>> {
        Some(other)
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
pub trait Shape: 'static + Sized + Clone + Copy + fmt::Debug + AsIndex {
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
pub trait FromShape<O: Shape>: Shape<Index: FromIndex<O::Index>> {
    // Required methods

    /// Convert to this index from the given compile-time compatible index,
    /// returning None if they turn out not to be compatible at runtime.
    fn try_from_shape(other: O) -> Option<Self>;

    // Provided methods

    /// Convert to this index from the given compile-time compatible index,
    /// panicking if they turn out not to be compatible at runtime.
    fn from_shape_fail(other: O) -> Self {
        Self::try_from_shape(other).unwrap_or_else(|| {
            panic!(
                "Shapes are not compatible: type={} got={other:?}",
                std::any::type_name::<Self>()
            )
        })
    }
}

pub trait IntoShape<T: Shape>: Shape<Index: IntoIndex<T::Index>> {
    // Required methods

    /// Convert from this index into the given compile-time compatible index,
    /// returning None if they turn out not to be compatible at runtime.
    fn try_into_shape(self) -> Option<T>;

    // Provided methods

    /// Convert from this index into the given compile-time compatible index,
    /// panicking if they turn out not to be compatible at runtime.
    fn into_shape_fail(self) -> T;
}

impl<T: Shape, U: Shape> IntoShape<U> for T
where
    U: FromShape<T>,
{
    // Required methods

    fn try_into_shape(self) -> Option<U> {
        U::try_from_shape(self)
    }

    // Provided methods

    fn into_shape_fail(self) -> U {
        U::from_shape_fail(self)
    }
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

// TODO move this back into trait Index
/// Panics if the given index does not equal the given index.
/// This should probably only be used for comparing shapes
/// since the panic message uses that terminology
pub fn shape_mismatch_fail<S1: Shape, S2: IntoShape<S1>>(a: S1, b: S2) {
    let b: S1 = b.into_shape_fail();
    for (i, s) in a.as_index().into_iter().zip(b.as_index().into_iter()) {
        if i != s {
            panic!("Shapes do not match: expected={a:?} got={b:?}");
        }
    }
}

impl<const N: usize> AsIndex for [usize; N] {
    type Index = [usize; N];

    fn as_index(&self) -> Self::Index {
        *self
    }
}

impl<const N: usize> Shape for [usize; N] {}

impl<const N: usize> FromShape<[usize; N]> for [usize; N] {
    fn try_from_shape(other: [usize; N]) -> Option<[usize; N]> {
        Some(other)
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

impl<D1: Dim + FromDim<E1>, E1: Dim> FromShape<(E1,)> for (D1,) {
    fn try_from_shape(other: (E1,)) -> Option<(D1,)> {
        Some((other.0.try_into_dim()?,))
    }
}

impl<D1: Dim> FromShape<[usize; 1]> for (D1,) {
    fn try_from_shape(other: [usize; 1]) -> Option<(D1,)> {
        Some((other[0].try_into_dim()?,))
    }
}

impl<D1: Dim> FromShape<(D1,)> for [usize; 1] {
    fn try_from_shape(other: (D1,)) -> Option<[usize; 1]> {
        Some(other.as_index())
    }
}

impl<D1: Dim, D2: Dim> AsIndex for (D1, D2) {
    type Index = [usize; 2];

    fn as_index(&self) -> Self::Index {
        [self.0.into(), self.1.into()]
    }
}

impl<D1: Dim, D2: Dim> Shape for (D1, D2) {}

impl<D1: Dim + FromDim<E1>, D2: Dim + FromDim<E2>, E1: Dim, E2: Dim> FromShape<(E1, E2)>
    for (D1, D2)
{
    fn try_from_shape(other: (E1, E2)) -> Option<(D1, D2)> {
        Some((other.0.try_into_dim()?, other.1.try_into_dim()?))
    }
}

impl<D1: Dim, D2: Dim> FromShape<[usize; 2]> for (D1, D2) {
    fn try_from_shape(other: [usize; 2]) -> Option<(D1, D2)> {
        Some((other[0].try_into_dim()?, other[1].try_into_dim()?))
    }
}

impl<D1: Dim, D2: Dim> FromShape<(D1, D2)> for [usize; 2] {
    fn try_from_shape(other: (D1, D2)) -> Option<[usize; 2]> {
        Some(other.as_index())
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
pub struct View<'a, S: Shape, E, D> {
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
pub struct ViewMut<'a, S: Shape, E, D> {
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
pub trait IntoView<S: Shape> {
    type Element;
    type Data;

    fn view(&self) -> View<S, Self::Element, Self::Data>;
}

impl<S: FromShape<S2>, S2: Shape, E, D> IntoView<S> for Array<S2, E, D> {
    type Element = E;
    type Data = D;

    fn view(&self) -> View<S, Self::Element, Self::Data> {
        View {
            shape: self.shape.into_shape_fail(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.into_index_fail(),
            data: &self.data,
        }
    }
}

impl<S: FromShape<S2>, S2: Shape, E, D> IntoView<S> for View<'_, S2, E, D> {
    type Element = E;
    type Data = D;

    fn view(&self) -> View<S, Self::Element, Self::Data> {
        View {
            shape: self.shape.into_shape_fail(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.into_index_fail(),
            data: self.data,
        }
    }
}

impl<S: FromShape<S2>, S2: Shape, E, D> IntoView<S> for ViewMut<'_, S2, E, D> {
    type Element = E;
    type Data = D;

    fn view(&self) -> View<S, Self::Element, Self::Data> {
        View {
            shape: self.shape.into_shape_fail(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.into_index_fail(),
            data: self.data,
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
pub trait IntoViewMut<S: Shape>: IntoView<S> {
    fn view_mut(&mut self) -> ViewMut<'_, S, Self::Element, Self::Data>;
}

impl<S: FromShape<S2>, S2: Shape, E, D> IntoViewMut<S> for Array<S2, E, D> {
    fn view_mut(&mut self) -> ViewMut<'_, S, Self::Element, Self::Data> {
        ViewMut {
            shape: self.shape.into_shape_fail(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.into_index_fail(),
            data: &mut self.data,
        }
    }
}

impl<S: FromShape<S2>, S2: Shape, E, D> IntoViewMut<S> for ViewMut<'_, S2, E, D> {
    fn view_mut(&mut self) -> ViewMut<'_, S, Self::Element, Self::Data> {
        ViewMut {
            shape: self.shape.into_shape_fail(),
            element: marker::PhantomData,
            offset: self.offset,
            strides: self.strides.into_index_fail(),
            data: self.data,
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
/// This `increment` function can now accept `&mut Array` or `&mut View`.
pub trait IntoTarget<S: Shape> {
    type Element;
    type Data;

    type Output;
    type Target: OutTarget<S, Element = Self::Element, Data = Self::Data, Output = Self::Output>;

    fn build(self, shape: S) -> Self::Target;
}

pub trait OutTarget<S: Shape>: IntoViewMut<S> {
    type Output;
    fn output(self) -> Self::Output;
}

// TODO can we get rid of the explicit reflexivity?
impl<'a, S: FromShape<S> + FromShape<S2>, S2: Shape, E, D> IntoTarget<S>
    for &'a mut Array<S2, E, D>
{
    type Element = E;
    type Data = D;

    type Target = ViewMut<'a, S, E, D>;
    type Output = ();

    fn build(self, _shape: S) -> Self::Target {
        self.view_mut()
    }
}

// TODO can we get rid of the explicit reflexivity?
impl<'a, S: FromShape<S> + FromShape<S2>, S2: Shape, E, D> IntoTarget<S>
    for &'a mut ViewMut<'a, S2, E, D>
{
    type Element = E;
    type Data = D;

    type Target = ViewMut<'a, S, E, D>;
    type Output = ();

    fn build(self, _shape: S) -> Self::Target {
        self.view_mut()
    }
}

impl<'a, S: FromShape<S2>, S2: Shape, E, D> OutTarget<S> for ViewMut<'a, S2, E, D> {
    type Output = ();

    fn output(self) -> Self::Output {
        ()
    }
}

pub struct Alloc<E>(marker::PhantomData<E>);

pub fn alloc<E>() -> Alloc<E> {
    Alloc(marker::PhantomData)
}

// TODO get rid of symmetric bound?
impl<'a, S: FromShape<S>, E: Default + Clone> IntoTarget<S> for Alloc<E> {
    type Element = E;
    type Data = Vec<E>;

    type Target = Array<S, Self::Element, Self::Data>;
    type Output = Self::Target;

    fn build(self, shape: S) -> Self::Target {
        let shape = shape.into_shape_fail();
        Array {
            shape,
            strides: shape.default_strides(),
            element: marker::PhantomData,
            offset: 0,
            data: vec![E::default(); shape.num_elements()],
        }
    }
}

impl<S: FromShape<S2>, S2: Shape, E, D> OutTarget<S> for Array<S2, E, D> {
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
        alloc, Array, Const, DefiniteRange, FromShape, IntoIndex, IntoTarget, IntoView,
        IntoViewMut, OutTarget, Shape,
    };
    use std::marker::PhantomData;

    #[test]
    fn test_shapes() {
        <(Const<3>, Const<4>)>::from_shape_fail((Const::<3>, Const::<4>));
        <(Const<3>, Const<4>)>::from_shape_fail((Const::<3>, 4));
        <(Const<3>, Const<4>)>::from_shape_fail((3, 4));
        <(Const<3>, Const<4>)>::from_shape_fail([3, 4]);
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
            S: Shape,
            O: IntoTarget<S, Element = f32, Data: AsRef<[f32]> + AsMut<[f32]>>,
        >(
            c_m: &impl IntoView<S, Element = f32, Data: AsRef<[f32]>>,
            out: O,
        ) -> O::Output {
            let c_m = c_m.view();

            let mut target = out.build(c_m.shape);
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
                        b * c_m[j.into_index_fail()]
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

        let mut b = Array {
            shape: [2, 2],
            strides: [2, 2].default_strides(),
            element: PhantomData,
            offset: 0,
            data: vec![0.; 4],
        };

        //bernstein_coef(&a, &mut b.view_mut());
        //bernstein_coef(&a.view(), &mut b);

        // TODO get rid of this explicit generic?
        bernstein_coef::<(Const<2>, Const<2>), _>(&a, alloc());

        bernstein_coef::<(Const<2>, usize), _>(&a, &mut b);
    }
}
