use std::cmp;
use std::fmt;
use std::marker;
use std::ops;

/// Represents the length of one axis of multi-dimensional data.
/// This may be a `usize`, (e.g. 3), or
/// a compile-time constant (e.g. `Const<3>`)
pub trait Dim: 'static + Sized + Clone + Copy + fmt::Debug + Into<usize> + PartialEq {}

/// Represents an axis length known at compile time.
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

/// A trait indicating that a type can be viewed as an [Index].
///
/// This is a convenience trait that unifies [Shape] and `Index`
/// for situations where either is applicable, such as ranges.
///
/// An `Index` always as an in-memory representation,
/// so this will manifest any compile-time constants in the `Shape`.
///
/// ```
/// use nada::{Const, AsIndex};
///
/// let a = (Const::<3>, Const::<4>);
/// let b = [5, 6];
///
/// assert_eq!(a.as_index(), [3, 4]);
/// assert_eq!(b.as_index(), [5, 6]);
/// ```
///
pub trait AsIndex {
    type Index: Index;

    /// Express this value as an index
    fn as_index(&self) -> Self::Index;
}

/// Represents length of each axis of multi-dimensional data.
/// This may be a tuple of [Dim], i.e. `usize` / `Const<N>`,
/// or a fixed-length array of `usize`.
/// Each element corresponds to one axis length.
///
/// ```
/// // Several shape representations that are equivalent:
///
/// use nada::{Const, ShapeEq};
///
/// let s = (Const::<3>, Const::<4>);
///
/// assert!(s.shape_eq(&(Const::<3>, Const::<4>)));
/// assert!(s.shape_eq(&(Const::<3>, 4)));
/// assert!(s.shape_eq(&(3, 4)));
/// assert!(s.shape_eq(&[3, 4]));
/// ```
///
/// Since there are multiple equivalent `Shape` representations,
/// **no two shapes should be assumed to be the same type**,
/// even when expressing a "same shapes" type bound.
/// The correct way to handle this is using two generic Shape types
/// that are linked by the [ShapeEq] trait.
/// `ShapeEq` is not implemented for shapes that are guaranteed incompatible at compile-time,
/// so shape mismatches
///
/// ```
/// use nada::{Shape, ShapeEq, Const};
///
/// fn check_same_shapes<S1: Shape + ShapeEq<S2>, S2: Shape>(a: S1, b: S2) {
///     if a.shape_eq(&b) {
///         println!("{a:?} == {b:?}");
///     } else {
///         println!("{a:?} != {b:?}");
///     }
/// }
///
/// check_same_shapes((Const::<5>, 6), (Const::<5>, Const::<6>)); // Equal
/// check_same_shapes((Const::<5>, 7), (Const::<5>, Const::<6>)); // Not equal at runtime
/// //check_same_shapes((Const::<4>, 6), (Const::<5>, Const::<6>)); // Not equal at compile time (type error)
/// ```
pub trait Shape: 'static + Sized + Clone + Copy + fmt::Debug + AsIndex {
    // Provided methods

    /// How many total elements are contained within multi-dimensional data
    /// of this shape.
    ///
    /// ```
    /// use nada::Shape;
    ///
    /// assert_eq!([3, 4].num_elements(), 12)
    /// ```
    fn num_elements(&self) -> usize {
        self.as_index().into_iter().product()
    }

    /// The stride values for multi-dimensional data of this shape.
    /// This is assuming the data is stored in a C-contiguous fashion,
    /// where the first axis changes slowest as you traverse the data
    /// (i.e. has the largest stride.)
    ///
    /// ```
    /// use nada::Shape;
    ///
    /// assert_eq!([3, 4].default_strides(), [4, 1]);
    /// // moving one unit along the first axis requires a stride of 4 elements,
    /// // moving one unit along the second axis requires a stride of 1 element.
    /// ```
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
    ///
    /// ```
    /// use nada::Shape;
    ///
    /// let shape = [2, 6];
    /// shape.out_of_bounds_fail(&[1, 5]); // No panic, since 1 < 2 and 5 < 6
    /// ```
    fn out_of_bounds_fail(&self, idx: &Self::Index) {
        for (i, s) in idx.into_iter().zip(self.as_index().into_iter()) {
            if i >= s {
                panic!("Index out of bounds: index={idx:?} shape={self:?}");
            }
        }
    }

    /// Panics if the given shape does not equal this shape.
    ///
    /// Fails to compile if the two shape types could never hold equal values,
    /// such as `(Const::<2>,)` and `(Const::<3>,)`,
    /// or `[2, 3]` and `[2, 3, 4]`
    /// (see [ShapeEq])
    ///
    /// ```
    /// use nada::{Const, Shape};
    ///
    /// let shape1 = [2, 3];
    /// let shape2 = (Const::<2>, Const::<3>);
    /// shape1.shape_mismatch_fail(&shape2); // No panic, since the shapes are the same
    /// ```
    fn shape_mismatch_fail<S: Shape>(&self, other: &S)
    where
        Self: ShapeEq<S>,
    {
        if !self.shape_eq(other) {
            panic!("Shapes do not match: {self:?} != {other:?}");
        }
    }
}

/// Represents the coordinates of an element in multi-dimensional data,
/// e.g. a 2D index might be represented as `[usize; 2]`.
/// Each element corresponds to the index along that axis.
///
/// Unlike [Shape], the coordinates of an `Index` are always type `usize`.
/// Every `Shape` type has a corresponding `Index` type
pub trait Index:
    'static
    + Sized
    + Clone
    + Copy
    + fmt::Debug
    + ops::IndexMut<usize>
    + IntoIterator<Item = usize, IntoIter: DoubleEndedIterator + ExactSizeIterator>
    + AsIndex<Index = Self>
{
    // Required methods

    // TODO I don't think this method is well-formed for dynamic number of dimensions
    fn zero() -> Self;

    /// Converts this multi-dimensional index into a linear index
    /// by multiplying each coordinate by the corresponding stride.
    /// No bounds checking is performed.
    ///
    /// ```
    /// use nada::Index;
    ///
    /// let strides = [4, 1];
    /// assert_eq!([3, 5].to_i(&strides), 3 * 4 + 5 * 1);
    /// ```
    fn to_i(&self, strides: &Self) -> usize {
        strides
            .into_iter()
            .zip(self.into_iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Returns an iterator over the components of this index,
    /// that allows modifying the components.
    ///
    /// For non-mutable iteration, use `.into_iter()`.
    ///
    /// ```
    /// use nada::Index;
    ///
    /// let mut i = [3, 5];
    ///
    /// for e in i.iter_mut() {
    ///     *e *= 2;
    /// }
    ///
    /// assert_eq!(i, [6, 10]);
    /// ```
    // TODO use into_iter for mutable iteration too?
    fn iter_mut<'a>(&'a mut self) -> std::slice::IterMut<'a, usize>;
}

/// This [Shape] is equal to another `Shape`.
/// This trait is implemented for all pairs of shapes that have some chance of success,
/// e.g. `[usize; 1] -> (Const<3>,)` (at runtime, the usize is checked to be 3.)
/// It is not implemented for conversions that have no chance of success at compile time
/// e.g. `(Const<4>, Const<5>) -> (Const<4>, Const<6>)`
pub trait ShapeEq<O: Shape>: Shape<Index: IntoIndex<O::Index>> {
    /// Return `true` if the shapes are equal, `false` if they are not.
    fn shape_eq(&self, other: &O) -> bool;
}

/// This [Index] type can be converted from another index.
/// This trait is implemented for all conversions that have some chance of success.
/// It is not implemented for conversions that have no chance of success at compile time
/// e.g. `[usize; 2] -> [usize; 3]`
///
/// If two [Shapes](Shape) are equal, then their indices are compatible
/// and these conversions will never fail
/// (although you still must perform the type conversion with `from_index_fail`.)
///
/// See also: [IntoIndex].
pub trait FromIndex<O: Index>: Index {
    /// Convert to the given index into this compatible index type,
    /// returning None if they turn out not to be compatible at runtime.
    fn try_from_index(other: O) -> Option<Self>;

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

/// This [Index] can be converted into another index type.
///
/// This trait is implemented for all conversions that have some chance of success.
/// It is not implemented for conversions that have no chance of success at compile time
/// e.g. `[usize; 2] -> [usize; 3]`
///
/// If two [Shapes](Shape) are equal, then their indices are compatible
/// and these conversions will never fail
/// (although you still must perform the type conversion with `into_index_fail`.)
///
/// If you are implementing this trait, prefer [FromIndex] since
/// `IntoIndex` has a blanket implementation for `FromIndex`.
pub trait IntoIndex<T: Index>: Index {
    /// Convert from this index into the given compile-time compatible index,
    /// returning None if they turn out not to be compatible at runtime.
    fn try_into_index(self) -> Option<T>;

    /// Convert from this index into the given compile-time compatible index,
    /// panicking if they turn out not to be compatible at runtime.
    fn into_index_fail(self) -> T;
}

impl<T: Index, U> IntoIndex<U> for T
where
    U: FromIndex<T>,
{
    fn try_into_index(self) -> Option<U> {
        U::try_from_index(self)
    }

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

impl<const N: usize> ShapeEq<[usize; N]> for [usize; N] {
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

impl<D1: Dim + cmp::PartialEq<E1>, E1: Dim> ShapeEq<(E1,)> for (D1,) {
    fn shape_eq(&self, other: &(E1,)) -> bool {
        self.0 == other.0
    }
}

impl<D1: Dim + cmp::PartialEq<usize>> ShapeEq<[usize; 1]> for (D1,) {
    fn shape_eq(&self, other: &[usize; 1]) -> bool {
        self.0 == other[0]
    }
}

impl<D1: Dim> ShapeEq<(D1,)> for [usize; 1]
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

impl<D1: Dim + PartialEq<E1>, D2: Dim + PartialEq<E2>, E1: Dim, E2: Dim> ShapeEq<(E1, E2)>
    for (D1, D2)
{
    fn shape_eq(&self, other: &(E1, E2)) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<D1: Dim + PartialEq<usize>, D2: Dim + PartialEq<usize>> ShapeEq<[usize; 2]> for (D1, D2) {
    fn shape_eq(&self, other: &[usize; 2]) -> bool {
        self.0 == other[0] && self.1 == other[1]
    }
}

impl<D1: Dim, D2: Dim> ShapeEq<(D1, D2)> for [usize; 2]
where
    usize: PartialEq<D1>,
    usize: PartialEq<D2>,
{
    fn shape_eq(&self, other: &(D1, D2)) -> bool {
        self[0] == other.0 && self[1] == other.1
    }
}

/////////////////////////////////////////////

// TODO Get rid of DefiniteRange trait
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

/// A multi-dimensional array.
///
/// This type owns the underlying data.
/// For a non-owning types, see [View] and [ViewMut],
#[derive(Debug, Clone)]
pub struct Array<S: Shape, D> {
    shape: S,
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
pub struct View<'a, S: Shape, D: ?Sized> {
    shape: S,
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
pub struct ViewMut<'a, S: Shape, D: ?Sized> {
    shape: S,
    offset: usize,
    strides: S::Index,
    data: &'a mut D,
}

/// This trait marks anything which can represent a read-only view into multi-dimensional data,
/// with the given shape.
///
/// Using this trait allows functions to accept any kind of read-only multi-dimensional data
/// with a given shape. For example:
/// ```
/// use nada::{IntoViewWithShape, Const};
///
/// // This function requires a 2x2 input
/// fn det2(a: &impl IntoViewWithShape<(Const<2>, Const<2>), [f32]>) -> f32 {
///     let a = a.view_with_shape((Const, Const)); // Convert to concrete type View
///
///     a[[0,0]] * a[[1,1]] - a[[0, 1]] * a[[1,0]]
/// }
/// ```
///
/// This `det2` function can now accept `&Array`, `&View`, or `&ViewMut`.
pub trait IntoViewWithShape<S: Shape, D: ?Sized> {
    fn view_with_shape(&self, shape: S) -> View<S, D>;
}

/// This trait marks anything which can represent a read-only view into multi-dimensional data,
/// such as:
/// * [Array]
/// * [View]
/// * [ViewMut]
///
/// Using this trait allows functions to accept any kind of read-only multi-dimensional data. For example:
/// ```
/// use nada::IntoView;
///
/// // This function can take any dimension input
/// fn sum(a: &impl IntoView<[f32]>) -> f32 {
///     let a = a.view(); // Convert to concrete type View
///
///     // Iterate and sum
///     a.into_iter().sum()
/// }
/// ```
///
/// This `sum` function can now accept `&Array`, `&View`, or `&ViewMut`.
pub trait IntoView<D: ?Sized>: IntoViewWithShape<Self::NativeShape, D> {
    // The native shape type of this data
    type NativeShape: Shape;

    fn view(&self) -> View<Self::NativeShape, D>;
}

impl<S: Shape, S2: Shape + ShapeEq<S>, D: ?Sized, D2: ops::Deref<Target = D>>
    IntoViewWithShape<S, D> for Array<S2, D2>
{
    fn view_with_shape(&self, shape: S) -> View<S, D> {
        self.shape.shape_mismatch_fail(&shape);
        View {
            shape,
            offset: self.offset,
            strides: self.strides.into_index_fail(),
            data: &self.data,
        }
    }
}

impl<S: Shape + ShapeEq<S>, D: ?Sized, D2: ops::Deref<Target = D>> IntoView<D> for Array<S, D2> {
    type NativeShape = S;

    fn view(&self) -> View<S, D> {
        View {
            shape: self.shape.clone(),
            offset: self.offset,
            strides: self.strides.clone(),
            data: &self.data,
        }
    }
}

impl<S: Shape, S2: Shape + ShapeEq<S>, D: ?Sized> IntoViewWithShape<S, D> for View<'_, S2, D> {
    fn view_with_shape(&self, shape: S) -> View<S, D> {
        self.shape.shape_mismatch_fail(&shape);
        View {
            shape,
            offset: self.offset,
            strides: self.strides.into_index_fail(),
            data: self.data,
        }
    }
}

impl<S: Shape + ShapeEq<S>, D: ?Sized> IntoView<D> for View<'_, S, D> {
    type NativeShape = S;

    fn view(&self) -> View<S, D> {
        View {
            shape: self.shape.clone(),
            offset: self.offset,
            strides: self.strides.clone(),
            data: self.data,
        }
    }
}

impl<S: Shape, S2: Shape + ShapeEq<S>, D: ?Sized> IntoViewWithShape<S, D> for ViewMut<'_, S2, D> {
    fn view_with_shape(&self, shape: S) -> View<S, D> {
        self.shape.shape_mismatch_fail(&shape);
        View {
            shape,
            offset: self.offset,
            strides: self.strides.into_index_fail(),
            data: self.data,
        }
    }
}

impl<S: Shape + ShapeEq<S>, D: ?Sized> IntoView<D> for ViewMut<'_, S, D> {
    type NativeShape = S;

    fn view(&self) -> View<S, D> {
        View {
            shape: self.shape.clone(),
            offset: self.offset,
            strides: self.strides.clone(),
            data: self.data,
        }
    }
}

/// This trait marks anything which can represent a mutable view into multi-dimensional data,
/// with the given shape.
///
/// Using this trait allows functions to accept any kind of mutable multi-dimensional data
/// with a given shape. For example:
/// ```
/// use nada::{IntoViewMutWithShape, Const};
///
/// // This function requires a 2x2 input
/// fn transpose_assign2(a: &mut impl IntoViewMutWithShape<(Const<2>, Const<2>), [f32]>) {
///     let mut a = a.view_mut_with_shape((Const, Const)); // Convert to concrete type View
///
///     a[[0, 1]] = a[[1,0]];
/// }
/// ```
///
/// This `transpose_assign2` function can now accept `&mut Array` or `&ViewMut`.
pub trait IntoViewMutWithShape<S: Shape, D: ?Sized>: IntoViewWithShape<S, D> {
    fn view_mut_with_shape(&mut self, shape: S) -> ViewMut<'_, S, D>;
}

/// This trait marks anything which can represent a mutable view into multi-dimensional data,
/// such as:
/// * [Array]
/// * [ViewMut]
///
/// Using this trait allows functions to accept any kind of mutable multi-dimensional data. For example:
/// ```
/// use nada::IntoViewMut;
///
/// // This function can take any dimension input
/// fn increment(a: &mut impl IntoViewMut<[f32]>) {
///     let mut a = a.view_mut(); // Convert to concrete type View
///
///     // Iterate and sum
///     for e in &mut a {
///         *e += 1.0;
///     }
/// }
/// ```
///
/// This `increment` function can now accept `&mut Array` or `&mut ViewMut`.
pub trait IntoViewMut<D: ?Sized>: IntoView<D> + IntoViewMutWithShape<Self::NativeShape, D> {
    fn view_mut(&mut self) -> ViewMut<Self::NativeShape, D>;
}

impl<
        S: Shape,
        S2: Shape + ShapeEq<S>,
        D: ?Sized,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > IntoViewMutWithShape<S, D> for Array<S2, D2>
{
    fn view_mut_with_shape(&mut self, shape: S) -> ViewMut<'_, S, D> {
        self.shape.shape_mismatch_fail(&shape);
        ViewMut {
            shape,
            offset: self.offset,
            strides: self.strides.into_index_fail(), // Shouldn't fail if shapes match
            data: &mut self.data,
        }
    }
}

impl<S: Shape + ShapeEq<S>, D: ?Sized, D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>>
    IntoViewMut<D> for Array<S, D2>
{
    fn view_mut(&mut self) -> ViewMut<'_, S, D> {
        ViewMut {
            shape: self.shape.clone(),
            offset: self.offset,
            strides: self.strides.clone(),
            data: &mut self.data,
        }
    }
}

impl<S: Shape, S2: Shape + ShapeEq<S>, D: ?Sized> IntoViewMutWithShape<S, D>
    for ViewMut<'_, S2, D>
{
    fn view_mut_with_shape(&mut self, shape: S) -> ViewMut<'_, S, D> {
        self.shape.shape_mismatch_fail(&shape);
        ViewMut {
            shape,
            offset: self.offset,
            strides: self.strides.into_index_fail(), // Shouldn't fail if shapes match
            data: self.data,
        }
    }
}

impl<S: Shape + ShapeEq<S>, D: ?Sized> IntoViewMut<D> for ViewMut<'_, S, D> {
    fn view_mut(&mut self) -> ViewMut<'_, S, D> {
        ViewMut {
            shape: self.shape.clone(),
            offset: self.offset,
            strides: self.strides.clone(),
            data: self.data,
        }
    }
}

/////////////////////////////////////////////

impl<S: Shape, E> View<'_, S, [E]> {
    pub unsafe fn get_unchecked(&self, idx: S::Index) -> &E {
        self.data
            .get_unchecked(self.offset + idx.to_i(&self.strides))
    }
}

impl<S: Shape, E> ops::Index<S::Index> for View<'_, S, [E]> {
    type Output = E;
    fn index(&self, idx: S::Index) -> &E {
        self.shape.out_of_bounds_fail(&idx);
        unsafe { self.get_unchecked(idx) }
    }
}
impl<'a, S: Shape, E> IntoIterator for &'a View<'_, S, [E]> {
    type Item = &'a E;
    type IntoIter = NdIter<'a, S, E>;
    fn into_iter(self) -> Self::IntoIter {
        NdIter {
            shape: self.shape.clone(),
            strides: self.strides,
            data: &self.data[self.offset..],
            idx: (..self.shape).first(),
        }
    }
}
impl<S: Shape, E> ViewMut<'_, S, [E]> {
    pub unsafe fn get_unchecked(&self, idx: S::Index) -> &E {
        self.data
            .get_unchecked(self.offset + idx.to_i(&self.strides))
    }
}
impl<S: Shape, E> ops::Index<S::Index> for ViewMut<'_, S, [E]> {
    type Output = E;
    fn index(&self, idx: S::Index) -> &E {
        self.shape.out_of_bounds_fail(&idx);
        unsafe { self.get_unchecked(idx) }
    }
}
impl<'a, S: Shape, E> IntoIterator for &'a ViewMut<'_, S, [E]> {
    type Item = &'a E;
    type IntoIter = NdIter<'a, S, E>;
    fn into_iter(self) -> Self::IntoIter {
        NdIter {
            shape: self.shape.clone(),
            strides: self.strides,
            data: &self.data[self.offset..],
            idx: (..self.shape).first(),
        }
    }
}
impl<S: Shape, E, D: ops::Deref<Target = [E]>> Array<S, D> {
    pub unsafe fn get_unchecked(&self, idx: S::Index) -> &E {
        self.data
            .as_ref()
            .get_unchecked(self.offset + idx.to_i(&self.strides))
    }
}
impl<S: Shape, E, D: ops::Deref<Target = [E]>> ops::Index<S::Index> for Array<S, D> {
    type Output = E;
    fn index(&self, idx: S::Index) -> &E {
        self.shape.out_of_bounds_fail(&idx);
        unsafe { self.get_unchecked(idx) }
    }
}
impl<'a, S: Shape, E: 'a, D: ops::Deref<Target = [E]>> IntoIterator for &'a Array<S, D> {
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

impl<S: Shape, E> ViewMut<'_, S, [E]> {
    pub unsafe fn get_unchecked_mut(&mut self, idx: S::Index) -> &mut E {
        self.data
            .get_unchecked_mut(self.offset + idx.to_i(&self.strides))
    }
}
impl<S: Shape, E> ops::IndexMut<S::Index> for ViewMut<'_, S, [E]> {
    fn index_mut(&mut self, idx: S::Index) -> &mut E {
        self.shape.out_of_bounds_fail(&idx);
        unsafe { self.get_unchecked_mut(idx) }
    }
}
impl<'a, S: Shape, E> IntoIterator for &'a mut ViewMut<'_, S, [E]> {
    type Item = &'a mut E;
    type IntoIter = NdIterMut<'a, S, E>;
    fn into_iter(self) -> Self::IntoIter {
        NdIterMut {
            shape: self.shape.clone(),
            strides: self.strides,
            data: &mut self.data[self.offset..],
            idx: (..self.shape).first(),
        }
    }
}
impl<S: Shape, E, D: ops::Deref<Target = [E]> + ops::DerefMut<Target = [E]>> Array<S, D> {
    pub unsafe fn get_unchecked_mut(&mut self, idx: S::Index) -> &mut E {
        self.data
            .as_mut()
            .get_unchecked_mut(self.offset + idx.to_i(&self.strides))
    }
}
impl<S: Shape, E, D: ops::Deref<Target = [E]> + ops::DerefMut<Target = [E]>> ops::IndexMut<S::Index>
    for Array<S, D>
{
    fn index_mut(&mut self, idx: S::Index) -> &mut E {
        self.shape.out_of_bounds_fail(&idx);
        unsafe { self.get_unchecked_mut(idx) }
    }
}
impl<'a, S: Shape, E: 'a, D: ops::DerefMut<Target = [E]>> IntoIterator for &'a mut Array<S, D> {
    type Item = &'a mut E;
    type IntoIter = NdIterMut<'a, S, E>;
    fn into_iter(self) -> Self::IntoIter {
        NdIterMut {
            shape: self.shape.clone(),
            strides: self.strides,
            data: &mut self.data[self.offset..],
            idx: (..self.shape).first(),
        }
    }
}

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

/// This trait marks anything which can serve as an output target for multi-dimensional data
/// with the given shape.
///
/// Using this trait allows functions to store their results in either an existing array
/// or a newly allocated array, at the discretion of the caller.
///
/// Here is an example:
/// ```
/// use nada::{IntoTargetWithShape, OutTarget, IntoViewMut, Const, alloc};
///
/// // This function requires a 2x2 input
/// fn eye2<O: IntoTargetWithShape<(Const<2>, Const<2>), [f32]>>(
///     out: O,
/// ) -> <O::Target as OutTarget>::Output {
///     let mut target = out.build_shape((Const, Const)); // Perform allocation, or no-op for existing data.
///     let mut a = target.view_mut(); // Get a mutable view of the output target
///
///     for ([i, j], e) in (&mut a).into_iter().nd_enumerate() {
///         *e = if i == j { 1.0 } else { 0.0 };
///     }
///
///     target.output() // Return the allocated array, or () for existing data
/// }
///
/// // Ask the function to allocate the output:
/// let mut a = eye2(alloc());
///
/// // Or store the output in an existing array:
/// eye2(&mut a);
/// ```
pub trait IntoTargetWithShape<S: Shape, D: ?Sized> {
    type Target: OutTarget + IntoViewMut<D, NativeShape = S>;

    fn build_shape(self, shape: S) -> Self::Target;
}

/// This trait marks anything which can serve as an output target for multi-dimensional data.
///
/// Using this trait allows functions to store their results in either an existing array
/// or a newly allocated array, at the discretion of the caller.
///
/// Here is an example:
/// ```
/// use nada::{IntoTarget, OutTarget, IntoViewMut, Const, alloc_shape};
///
/// fn ones<O: IntoTarget<[f32]>>(
///     out: O,
/// ) -> <O::Target as OutTarget>::Output {
///     let mut target = out.build(); // Perform allocation, or no-op for existing data.
///     let mut a = target.view_mut(); // Get a mutable view of the output target
///
///     for e in &mut a {
///         *e = 1.0;
///     }
///
///     target.output() // Return the allocated array, or () for existing data
/// }
///
/// // Ask the function to allocate the output:
/// let mut a = ones(alloc_shape((Const::<5>, Const::<6>)));
///
/// // Or fill an existing array:
/// ones(&mut a);
/// ```
///
/// To allocate the output, `alloc_shape()` is required instead of `alloc()`,
/// because `alloc()` provides no shape information
/// and none is supplied to `build()`.
///
/// If shape information can be supplied to `build()`,
/// consider using [IntoTargetWithShape] and `build_with_shape()` instead.
pub trait IntoTarget<D: ?Sized>: IntoTargetWithShape<Self::NativeShape, D> {
    type NativeShape: Shape;

    fn build(self) -> Self::Target;
}

/// The inetermediate representation of output data.
/// This generally serves two purposes:
/// * Providing a `.view_mut()` method to which data can be written
/// * Providing a `.output()` method to consume it and produce a return value
pub trait OutTarget {
    type Output;

    /// Consume this output target and produce its canonical return value
    /// (e.g. `()` for [ViewMut] or [Array] for [ArrayTarget])
    fn output(self) -> Self::Output;
}

impl<
        'a,
        S: Shape + ShapeEq<S>,
        S2: Shape + ShapeEq<S>,
        D: 'a + ?Sized,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > IntoTargetWithShape<S, D> for &'a mut Array<S2, D2>
{
    type Target = ViewMut<'a, S, D>;

    fn build_shape(self, shape: S) -> Self::Target {
        self.view_mut_with_shape(shape)
    }
}

impl<
        'a,
        S: Shape + ShapeEq<S>,
        D: 'a + ?Sized,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > IntoTarget<D> for &'a mut Array<S, D2>
{
    type NativeShape = S;

    fn build(self) -> Self::Target {
        self.view_mut()
    }
}

impl<'a, S: Shape + ShapeEq<S>, S2: Shape + ShapeEq<S>, D: ?Sized> IntoTargetWithShape<S, D>
    for &'a mut ViewMut<'a, S2, D>
{
    type Target = ViewMut<'a, S, D>;

    fn build_shape(self, shape: S) -> Self::Target {
        self.view_mut_with_shape(shape)
    }
}

impl<'a, S: Shape + ShapeEq<S>, D: ?Sized> IntoTarget<D> for &'a mut ViewMut<'a, S, D> {
    type NativeShape = S;

    fn build(self) -> Self::Target {
        self.view_mut()
    }
}

impl<'a, S: Shape, D: ?Sized> OutTarget for ViewMut<'a, S, D> {
    type Output = ();

    fn output(self) -> Self::Output {
        ()
    }
}

pub struct AllocShape<S: Shape, E> {
    shape: S,
    element: marker::PhantomData<E>,
}

pub fn alloc_shape<S: Shape, E>(shape: S) -> AllocShape<S, E> {
    AllocShape {
        shape,
        element: marker::PhantomData,
    }
}

/// An output target wrapping an owned [Array],
/// that when viewed has a different (but equal) shape
/// than the underlying `Array`.
///
/// This should never need to be constructed directly
/// (see [IntoTarget])
pub struct ArrayTarget<S: Shape, S2: Shape + ShapeEq<S>, D> {
    array: Array<S2, D>,
    shape: S,
}

impl<
        S: Shape,
        S2: Shape + ShapeEq<S>,
        S3: Shape + ShapeEq<S2>,
        D: ?Sized,
        D2: ops::Deref<Target = D>,
    > IntoViewWithShape<S, D> for ArrayTarget<S2, S3, D2>
{
    fn view_with_shape(&self, shape: S) -> View<S, D> {
        self.shape.shape_mismatch_fail(&shape);
        View {
            shape,
            offset: self.array.offset,
            strides: self.array.strides.into_index_fail().into_index_fail(),
            data: &self.array.data,
        }
    }
}

impl<S: Shape + ShapeEq<S>, S2: Shape + ShapeEq<S>, D: ?Sized, D2: ops::Deref<Target = D>>
    IntoView<D> for ArrayTarget<S, S2, D2>
{
    type NativeShape = S;

    fn view(&self) -> View<S, D> {
        View {
            shape: self.shape.clone(),
            offset: self.array.offset,
            strides: self.array.strides.into_index_fail(),
            data: &self.array.data,
        }
    }
}

impl<
        S: Shape,
        S2: Shape + ShapeEq<S>,
        S3: Shape + ShapeEq<S2>,
        D: ?Sized,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > IntoViewMutWithShape<S, D> for ArrayTarget<S2, S3, D2>
{
    fn view_mut_with_shape(&mut self, shape: S) -> ViewMut<'_, S, D> {
        self.shape.shape_mismatch_fail(&shape);
        ViewMut {
            shape,
            offset: self.array.offset,
            strides: self.array.strides.into_index_fail().into_index_fail(),
            data: &mut self.array.data,
        }
    }
}

impl<
        S: Shape + ShapeEq<S>,
        S2: Shape + ShapeEq<S>,
        D: ?Sized,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > IntoViewMut<D> for ArrayTarget<S, S2, D2>
{
    fn view_mut(&mut self) -> ViewMut<'_, S, D> {
        ViewMut {
            shape: self.shape.clone(),
            offset: self.array.offset,
            strides: self.array.strides.into_index_fail(),
            data: &mut self.array.data,
        }
    }
}

impl<'a, S: Shape, S2: Shape + ShapeEq<S>, D> OutTarget for ArrayTarget<S, S2, D> {
    type Output = Array<S2, D>;

    fn output(self) -> Self::Output {
        self.array
    }
}

impl<'a, S: Shape + ShapeEq<S>, S2: Shape + ShapeEq<S>, E: Default + Clone>
    IntoTargetWithShape<S, [E]> for AllocShape<S2, E>
{
    type Target = ArrayTarget<S, S2, Vec<E>>;

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
                offset: 0,
                data: vec![E::default(); self_shape.num_elements()],
            },
        }
    }
}

impl<'a, S: Shape + ShapeEq<S>, E: Default + Clone> IntoTarget<[E]> for AllocShape<S, E> {
    type NativeShape = S;

    fn build(self) -> Self::Target {
        let AllocShape { shape, .. } = self;
        ArrayTarget {
            shape,
            array: Array {
                shape,
                strides: shape.default_strides(),
                offset: 0,
                data: vec![E::default(); shape.num_elements()],
            },
        }
    }
}

pub struct Alloc<E> {
    element: marker::PhantomData<E>,
}

pub fn alloc<E>() -> Alloc<E> {
    Alloc {
        element: marker::PhantomData,
    }
}

impl<'a, S: Shape + ShapeEq<S>, E: Default + Clone, D: ?Sized> IntoTargetWithShape<S, D>
    for Alloc<E>
where
    Vec<E>: ops::DerefMut<Target = D> + ops::Deref<Target = D>,
{
    type Target = Array<S, Vec<E>>;

    fn build_shape(self, shape: S) -> Self::Target {
        Array {
            shape,
            strides: shape.default_strides(),
            offset: 0,
            data: vec![E::default(); shape.num_elements()],
        }
    }
}

impl<S: Shape, D> OutTarget for Array<S, D> {
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
        alloc, alloc_shape, Array, AsIndex, Const, DefiniteRange, IntoTarget, IntoTargetWithShape,
        IntoView, IntoViewMut, OutTarget, Shape, ShapeEq,
    };

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
    fn test_iter() {
        let mut t = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
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

        fn bernstein_coef<V: IntoView<[f32]>, O: IntoTargetWithShape<V::NativeShape, [f32]>>(
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

        dbg!(bernstein_coef(&a, alloc()));
        dbg!(bernstein_coef(&a, alloc_shape([2, 2])));
        panic!();

        //bernstein_coef(&a, &mut b);
    }

    #[test]
    fn test_sum() {
        fn sum(in1: &impl IntoView<[f32]>) -> f32 {
            let in1 = in1.view();

            in1.into_iter().sum()
        }

        let a = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
            offset: 0,
            data: vec![1., 2., 3., 4.],
        };

        sum(&a);
    }

    #[test]
    fn test_ones() {
        fn ones<O: IntoTarget<[f32]>>(out: O) -> <O::Target as OutTarget>::Output {
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
            offset: 0,
            data: vec![1., 2., 3., 4.],
        };

        ones(&mut a);
        ones(alloc_shape([4, 4]));
    }
}
