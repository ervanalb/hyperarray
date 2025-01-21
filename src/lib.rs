use std::fmt;
use std::marker;
use std::ops;

/*

// Helper trait for building up fixed-length arrays
pub trait FixedLengthArray: Sized {
    type OneBigger: FixedLengthArraySplit<Self>;

    fn prepend(self, e: usize) -> Self::OneBigger;
}

pub trait FixedLengthArraySplit<T> {
    fn split_first(self) -> (usize, T);
}

impl FixedLengthArray for [usize; 0] {
    type OneBigger = [usize; 1];

    fn prepend(self, e: usize) -> Self::OneBigger {
        [e]
    }
}

impl FixedLengthArraySplit<[usize; 0]> for [usize; 1] {
    fn split_first(self) -> (usize, [usize; 0]) {
        (self[0], [])
    }
}

impl FixedLengthArray for [usize; 1] {
    type OneBigger = [usize; 2];

    fn prepend(self, e: usize) -> Self::OneBigger {
        [e, self[0]]
    }
}

impl FixedLengthArraySplit<[usize; 1]> for [usize; 2] {
    fn split_first(self) -> (usize, [usize; 1]) {
        (self[0], [self[1]])
    }
}

impl FixedLengthArray for [usize; 2] {
    type OneBigger = [usize; 3];

    fn prepend(self, e: usize) -> Self::OneBigger {
        [e, self[0], self[1]]
    }
}

impl FixedLengthArraySplit<[usize; 2]> for [usize; 3] {
    fn split_first(self) -> (usize, [usize; 2]) {
        (self[0], [self[1], self[2]])
    }
}

*/

/// Represents an axis of unit length that can be broadcast to any size.
#[derive(Default, Clone, Copy)]
pub struct NewAxis;

impl fmt::Debug for NewAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_")
    }
}

/// Represents an axis length known at compile time.
/// The primitive type `usize` is used for a dimension not known at compile time,
/// or [NewAxis] for a unit-length broadcastable axis.
/// (Unlike with NumPy, an axis length `Const::<1>` or `1_usize` will not broadcast)
#[derive(Default, Clone, Copy)]
pub struct Const<const N: usize>;

impl<const N: usize> fmt::Debug for Const<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_const", N)
    }
}

pub trait BroadcastInto<T: Shape>: Shape {
    fn can_broadcast_into(self, other: T) -> bool;
    fn into_index(index: Self::Index) -> T::Index;
}

pub trait BroadcastIntoNoAlias<T: Shape>: BroadcastInto<T> {}

pub trait BroadcastTogether<T: Shape + BroadcastInto<Self::Output>>:
    Shape + BroadcastInto<Self::Output>
{
    type Output: Shape;

    fn broadcast_together(self, other: T) -> Option<Self::Output>;
}

// Marker trait for if this broadcast avoids aliasing
pub trait BroadcastShapeNoAlias<T: Shape + BroadcastInto<Self::Output>>:
    BroadcastTogether<T>
{
}

// Broadcast

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
pub trait Shape: 'static + Sized + Clone + Copy + AsIndex + fmt::Debug {
    // Provided methods

    /// How many total elements are contained within multi-dimensional data
    /// of this shape.
    ///
    /// ```
    /// use nada::Shape;
    ///
    /// assert_eq!((3, 4).num_elements(), 12)
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
    /// assert_eq!((3, 4).default_strides(), [4, 1]);
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
    /// let shape = (2, 6);
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
    /// let shape1 = (2, 3);
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

    /// Tries to broadcast the two shapes together, and panics if it fails.
    ///
    /// Fails to compile if the two shape types could never be broadcast together,
    /// such as `(Const::<2>,)` and `(Const::<3>,)`
    ///
    /// ```
    /// use nada::{Const, Shape};
    ///
    /// let shape1 = (Const::<2>, Const::<3>);
    /// let shape2 = (2, 3);
    /// let shape3 = (3,);
    /// let _b1 = shape1.broadcast_together_fail(shape2); // No panic since the shapes are equal
    /// let _b2 = shape1.broadcast_together_fail(shape3); // No panic since the shapes are compatible
    /// // TODO show the contents of b1 and b2
    /// ```
    fn broadcast_together_fail<S: Shape + BroadcastInto<Self::Output>>(
        self,
        other: S,
    ) -> Self::Output
    where
        Self: BroadcastTogether<S>,
    {
        self.broadcast_together(other).unwrap_or_else(|| {
            panic!("Shapes cannot be broadcast together: {self:?} and {other:?}");
        })
    }

    /// Tries to broadcast one shape into another, and panics if it fails.
    ///
    /// Fails to compile if the shape could never be broadcast into the given one,
    /// such as `(Const::<2>,)` and `(Const::<3>,)`
    ///
    /// ```
    /// use nada::{Const, Shape};
    ///
    /// let shape1 = (Const::<2>, Const::<3>);
    /// let shape2 = (2, 3);
    /// let shape3 = (10, 2, 3);
    /// let _b1 = shape1.broadcast_into_fail(shape2); // No panic since the shapes are equal
    /// let _b2 = shape1.broadcast_into_fail(shape3); // No panic since the shapes are compatible
    /// // TODO show the contents of b1 and b2
    /// ```
    fn broadcast_into_fail<S: Shape>(self, other: S)
    where
        Self: BroadcastInto<S>,
    {
        if !self.can_broadcast_into(other) {
            panic!("Shape {self:?} cannot be broadcast into shape {other:?}");
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
/// e.g. `(usize,) -> (Const<3>,)` (at runtime, the usize is checked to be 3.)
/// It is not implemented for conversions that have no chance of success at compile time
/// e.g. `(Const<4>, Const<5>) -> (Const<4>, Const<6>)`
pub trait ShapeEq<O: Shape>: Shape<Index: Into<O::Index>> {
    /// Return `true` if the shapes are equal, `false` if they are not.
    fn shape_eq(&self, other: &O) -> bool;
}

impl<const N: usize> AsIndex for [usize; N] {
    type Index = [usize; N];

    fn as_index(&self) -> Self::Index {
        *self
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

/////////////////////////////////////////////
// Private helper trait for fixed-length array building

pub trait FLA {
    type OneBigger: Split<OneSmaller = Self>;

    fn append(self, a: usize) -> Self::OneBigger;
}

pub trait Split {
    type OneSmaller: FLA<OneBigger = Self>;

    fn split(self) -> (Self::OneSmaller, usize);
}

impl FLA for [usize; 0] {
    type OneBigger = [usize; 1];

    fn append(self, a: usize) -> Self::OneBigger {
        [a]
    }
}

impl Split for [usize; 1] {
    type OneSmaller = [usize; 0];

    fn split(self) -> ([usize; 0], usize) {
        ([], self[0])
    }
}

impl FLA for [usize; 1] {
    type OneBigger = [usize; 2];

    fn append(self, a: usize) -> Self::OneBigger {
        [self[0], a]
    }
}

impl Split for [usize; 2] {
    type OneSmaller = [usize; 1];

    fn split(self) -> ([usize; 1], usize) {
        ([self[0]], self[1])
    }
}

impl FLA for [usize; 2] {
    type OneBigger = [usize; 3];

    fn append(self, a: usize) -> Self::OneBigger {
        [self[0], self[1], a]
    }
}

impl Split for [usize; 3] {
    type OneSmaller = [usize; 2];

    fn split(self) -> ([usize; 2], usize) {
        ([self[0], self[1]], self[2])
    }
}

/////////////////////////////////////////////
// Shape Implementations

// Shape

impl Shape for () {}

impl<S: Shape, const N: usize> Shape for (S, Const<N>) where S::Index: FLA<OneBigger: Index> {}

impl<S: Shape> Shape for (S, usize) where S::Index: FLA<OneBigger: Index> {}

impl<S: Shape> Shape for (S, NewAxis) {}

// AsIndex

impl AsIndex for () {
    type Index = [usize; 0];

    fn as_index(&self) -> Self::Index {
        []
    }
}

impl<S: AsIndex, const N: usize> AsIndex for (S, Const<N>)
where
    S::Index: FLA<OneBigger: Index>,
{
    type Index = <S::Index as FLA>::OneBigger;

    fn as_index(&self) -> Self::Index {
        self.0.as_index().append(N)
    }
}

impl<S: AsIndex> AsIndex for (S, usize)
where
    S::Index: FLA<OneBigger: Index>,
{
    type Index = <S::Index as FLA>::OneBigger;

    fn as_index(&self) -> Self::Index {
        self.0.as_index().append(self.1)
    }
}

impl<S: AsIndex> AsIndex for (S, NewAxis) {
    type Index = S::Index;

    fn as_index(&self) -> Self::Index {
        self.0.as_index()
    }
}

// BroadcastInto / BroadcastIntoNoAlias

impl BroadcastInto<()> for () {
    fn can_broadcast_into(self, _other: ()) -> bool {
        true
    }

    fn into_index(_index: [usize; 0]) -> [usize; 0] {
        []
    }
}
impl BroadcastIntoNoAlias<()> for () {}

impl<S1: BroadcastInto<S2>, S2: Shape> BroadcastInto<(S2, NewAxis)> for (S1, NewAxis) {
    fn can_broadcast_into(self, other: (S2, NewAxis)) -> bool {
        self.0.can_broadcast_into(other.0)
    }

    fn into_index(index: S1::Index) -> S2::Index {
        S1::into_index(index)
    }
}
impl<S1: BroadcastIntoNoAlias<S2>, S2: Shape> BroadcastIntoNoAlias<(S2, NewAxis)>
    for (S1, NewAxis)
{
}

impl<S2: Shape> BroadcastInto<(S2, NewAxis)> for ()
where
    (): BroadcastInto<S2>,
{
    fn can_broadcast_into(self, other: (S2, NewAxis)) -> bool {
        ().can_broadcast_into(other.0)
    }

    fn into_index(index: [usize; 0]) -> S2::Index {
        <() as BroadcastInto<S2>>::into_index(index)
    }
}
impl<S2: Shape> BroadcastIntoNoAlias<(S2, NewAxis)> for () where (): BroadcastInto<S2> {}

impl<S1: Shape + BroadcastInto<()>> BroadcastInto<()> for (S1, NewAxis) {
    fn can_broadcast_into(self, _other: ()) -> bool {
        self.0.can_broadcast_into(())
    }
    fn into_index(index: S1::Index) -> <() as AsIndex>::Index {
        S1::into_index(index)
    }
}
impl<S1: Shape + BroadcastIntoNoAlias<()>> BroadcastIntoNoAlias<()> for (S1, NewAxis) {}

impl<S1: BroadcastInto<S2>, S2: Shape, const N: usize> BroadcastInto<(S2, Const<N>)>
    for (S1, NewAxis)
where
    <S1 as AsIndex>::Index: FLA,
    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
{
    fn can_broadcast_into(self, other: (S2, Const<N>)) -> bool {
        self.0.can_broadcast_into(other.0)
    }
    fn into_index(index: <S1 as AsIndex>::Index) -> <<S2 as AsIndex>::Index as FLA>::OneBigger {
        S1::into_index(index).append(0)
    }
}
//impl<S1: BroadcastInto<S2>, S2: Shape, const N: usize> BroadcastIntoNoAlias<(S2, Const<N>)>
//    for (S1, NewAxis)
//where
//    <S1 as AsIndex>::Index: FLA,
//    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
//{ ! }

impl<S1: BroadcastInto<S2>, S2: Shape> BroadcastInto<(S2, usize)> for (S1, NewAxis)
where
    <S1 as AsIndex>::Index: FLA,
    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
{
    fn can_broadcast_into(self, other: (S2, usize)) -> bool {
        self.0.can_broadcast_into(other.0)
    }
    fn into_index(index: <S1 as AsIndex>::Index) -> <<S2 as AsIndex>::Index as FLA>::OneBigger {
        S1::into_index(index).append(0)
    }
}
//impl<S1: BroadcastIntoNoAlias<S2>, S2: Shape> BroadcastIntoNoAlias<(S2, usize)> for (S1, NewAxis)
//where
//    <S1 as AsIndex>::Index: FLA,
//    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
//{ ! }

impl<S2: Shape, const N: usize> BroadcastInto<(S2, Const<N>)> for ()
where
    (): BroadcastInto<S2>,
    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
{
    fn can_broadcast_into(self, other: (S2, Const<N>)) -> bool {
        ().can_broadcast_into(other.0)
    }
    fn into_index(index: [usize; 0]) -> <<S2 as AsIndex>::Index as FLA>::OneBigger {
        <() as BroadcastInto<S2>>::into_index(index).append(0)
    }
}
//impl<S2: Shape, const N: usize> BroadcastIntoNoAlias<(S2, Const<N>)> for ()
//where
//    (): BroadcastIntoNoAlias<S2>,
//    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
//{ ! }

impl<S2: Shape> BroadcastInto<(S2, usize)> for ()
where
    (): BroadcastInto<S2>,
    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
{
    fn can_broadcast_into(self, other: (S2, usize)) -> bool {
        ().can_broadcast_into(other.0)
    }
    fn into_index(index: [usize; 0]) -> <<S2 as AsIndex>::Index as FLA>::OneBigger {
        <() as BroadcastInto<S2>>::into_index(index).append(0)
    }
}
//impl<S2: Shape> BroadcastIntoNoAlias<(S2, usize)> for ()
//where
//    (): BroadcastIntoNoAlias<S2>,
//    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
//{ ! }

impl<S1: BroadcastInto<S2>, S2: Shape, const N: usize> BroadcastInto<(S2, Const<N>)>
    for (S1, Const<N>)
where
    <S1 as AsIndex>::Index: FLA<OneBigger: Index>,
    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
{
    fn can_broadcast_into(self, other: (S2, Const<N>)) -> bool {
        self.0.can_broadcast_into(other.0)
    }
    fn into_index(
        index: <<S1 as AsIndex>::Index as FLA>::OneBigger,
    ) -> <<S2 as AsIndex>::Index as FLA>::OneBigger {
        let (index_rest, index_last) = index.split();
        S1::into_index(index_rest).append(index_last)
    }
}
impl<S1: BroadcastIntoNoAlias<S2>, S2: Shape, const N: usize> BroadcastIntoNoAlias<(S2, Const<N>)>
    for (S1, Const<N>)
where
    <S1 as AsIndex>::Index: FLA<OneBigger: Index>,
    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
{
}

impl<S1: BroadcastInto<S2>, S2: Shape, const N: usize> BroadcastInto<(S2, Const<N>)> for (S1, usize)
where
    <S1 as AsIndex>::Index: FLA<OneBigger: Index>,
    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
{
    fn can_broadcast_into(self, other: (S2, Const<N>)) -> bool {
        self.0.can_broadcast_into(other.0) && self.1 == N
    }
    fn into_index(
        index: <<S1 as AsIndex>::Index as FLA>::OneBigger,
    ) -> <<S2 as AsIndex>::Index as FLA>::OneBigger {
        let (index_rest, index_last) = index.split();
        S1::into_index(index_rest).append(index_last)
    }
}
impl<S1: BroadcastIntoNoAlias<S2>, S2: Shape, const N: usize> BroadcastIntoNoAlias<(S2, Const<N>)>
    for (S1, usize)
where
    <S1 as AsIndex>::Index: FLA<OneBigger: Index>,
    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
{
}

impl<S1: BroadcastInto<S2>, S2: Shape, const N: usize> BroadcastInto<(S2, usize)> for (S1, Const<N>)
where
    <S1 as AsIndex>::Index: FLA<OneBigger: Index>,
    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
{
    fn can_broadcast_into(self, other: (S2, usize)) -> bool {
        self.0.can_broadcast_into(other.0) && N == other.1
    }
    fn into_index(
        index: <<S1 as AsIndex>::Index as FLA>::OneBigger,
    ) -> <<S2 as AsIndex>::Index as FLA>::OneBigger {
        let (index_rest, index_last) = index.split();
        S1::into_index(index_rest).append(index_last)
    }
}
impl<S1: BroadcastIntoNoAlias<S2>, S2: Shape, const N: usize> BroadcastIntoNoAlias<(S2, usize)>
    for (S1, Const<N>)
where
    <S1 as AsIndex>::Index: FLA<OneBigger: Index>,
    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
{
}

impl<S1: BroadcastInto<S2>, S2: Shape> BroadcastInto<(S2, usize)> for (S1, usize)
where
    <S1 as AsIndex>::Index: FLA<OneBigger: Index>,
    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
{
    fn can_broadcast_into(self, other: (S2, usize)) -> bool {
        self.0.can_broadcast_into(other.0) && self.1 == other.1
    }
    fn into_index(
        index: <<S1 as AsIndex>::Index as FLA>::OneBigger,
    ) -> <<S2 as AsIndex>::Index as FLA>::OneBigger {
        let (index_rest, index_last) = index.split();
        S1::into_index(index_rest).append(index_last)
    }
}
impl<S1: BroadcastIntoNoAlias<S2>, S2: Shape> BroadcastIntoNoAlias<(S2, usize)> for (S1, usize)
where
    <S1 as AsIndex>::Index: FLA<OneBigger: Index>,
    <S2 as AsIndex>::Index: FLA<OneBigger: Index>,
{
}

// ShapeEq

impl ShapeEq<()> for () {
    fn shape_eq(&self, _other: &()) -> bool {
        true
    }
}

impl<S2: Shape> ShapeEq<(S2, NewAxis)> for ()
where
    (): ShapeEq<S2>,
{
    fn shape_eq(&self, other: &(S2, NewAxis)) -> bool {
        ().shape_eq(&other.0)
    }
}

//impl<S2: Shape<Index=[usize; 0]>, const N: usize> ShapeEq<(S2, Const<N>)> for () { ! }

//impl<S2: Shape> ShapeEq<(S2, usize)> for () { ! }

impl<S1: ShapeEq<()>> ShapeEq<()> for (S1, NewAxis) {
    fn shape_eq(&self, _other: &()) -> bool {
        self.0.shape_eq(&())
    }
}

impl<S1: ShapeEq<S2>, S2: Shape> ShapeEq<(S2, NewAxis)> for (S1, NewAxis) {
    fn shape_eq(&self, other: &(S2, NewAxis)) -> bool {
        self.0.shape_eq(&other.0)
    }
}

//impl<S1: ShapeEq<S2>, S2: Shape, const N: usize> ShapeEq<(S2, Const<N>)> for (S1, NewAxis) { ! }

//impl<S1: ShapeEq<S2>, S2: Shape> ShapeEq<(S2, usize)> for (S1, NewAxis) { ! }

//impl<S1: ShapeEq<()>, const N: usize> ShapeEq<()> for (S1, Const<N>) { ! }

//impl<S1: ShapeEq<S2>, S2: Shape, const N: usize> ShapeEq<(S2, Const<N>)> for (S1, NewAxis) { ! }

impl<S1: ShapeEq<S2>, S2: Shape, const N: usize> ShapeEq<(S2, Const<N>)> for (S1, Const<N>)
where
    (S1, Const<N>): Shape<Index = <(S2, Const<N>) as AsIndex>::Index>,
    (S2, Const<N>): Shape,
{
    fn shape_eq(&self, other: &(S2, Const<N>)) -> bool {
        self.0.shape_eq(&other.0)
    }
}

impl<S1: ShapeEq<S2>, S2: Shape, const N: usize> ShapeEq<(S2, Const<N>)> for (S1, usize)
where
    (S1, usize): Shape<Index = <(S2, Const<N>) as AsIndex>::Index>,
    (S2, Const<N>): Shape,
{
    fn shape_eq(&self, other: &(S2, Const<N>)) -> bool {
        self.0.shape_eq(&other.0) && self.1 == N
    }
}

//impl<S1: ShapeEq<()>> ShapeEq<()> for (S1, usize) { ! }

// impl<S1: ShapeEq<S2>, S2: Shape, const N: usize> ShapeEq<(S2, usize)> for (S1, NewAxis) { ! }

impl<S1: ShapeEq<S2>, S2: Shape, const N: usize> ShapeEq<(S2, usize)> for (S1, Const<N>)
where
    (S1, Const<N>): Shape<Index = <(S2, usize) as AsIndex>::Index>,
    (S2, usize): Shape,
{
    fn shape_eq(&self, other: &(S2, usize)) -> bool {
        self.0.shape_eq(&other.0) && N == other.1
    }
}

impl<S1: ShapeEq<S2>, S2: Shape> ShapeEq<(S2, usize)> for (S1, usize)
where
    (S1, usize): Shape<Index = <(S2, usize) as AsIndex>::Index>,
    (S2, usize): Shape,
{
    fn shape_eq(&self, other: &(S2, usize)) -> bool {
        self.0.shape_eq(&other.0) && self.1 == other.1
    }
}

// BroadcastTogether / BroadcastTogetherNoAlias

impl BroadcastTogether<()> for () {
    type Output = ();

    fn broadcast_together(self, _other: ()) -> Option<Self::Output> {
        Some(())
    }
}

impl<S2: Shape> BroadcastTogether<(S2, NewAxis)> for ()
where
    (): BroadcastInto<(S2, NewAxis)>,
    (S2, NewAxis): BroadcastInto<(S2, NewAxis)>,
{
    type Output = (S2, NewAxis);

    fn broadcast_together(self, other: (S2, NewAxis)) -> Option<Self::Output> {
        Some(other)
    }
}

impl<S2: Shape, const N: usize> BroadcastTogether<(S2, Const<N>)> for ()
where
    (): BroadcastInto<(S2, Const<N>)>,
    (S2, Const<N>): BroadcastInto<(S2, Const<N>)>,
{
    type Output = (S2, Const<N>);

    fn broadcast_together(self, other: (S2, Const<N>)) -> Option<Self::Output> {
        Some(other)
    }
}

impl<S2: Shape> BroadcastTogether<(S2, usize)> for ()
where
    (): BroadcastInto<(S2, usize)>,
    (S2, usize): BroadcastInto<(S2, usize)>,
{
    type Output = (S2, usize);

    fn broadcast_together(self, other: (S2, usize)) -> Option<Self::Output> {
        Some(other)
    }
}

impl<S1: Shape> BroadcastTogether<()> for (S1, NewAxis)
where
    (S1, NewAxis): BroadcastInto<(S1, NewAxis)>,
    (): BroadcastInto<(S1, NewAxis)>,
{
    type Output = (S1, NewAxis);

    fn broadcast_together(self, _other: ()) -> Option<Self::Output> {
        Some(self)
    }
}

impl<S1: BroadcastTogether<S2>, S2: Shape> BroadcastTogether<(S2, NewAxis)> for (S1, NewAxis)
where
    (S1, NewAxis): BroadcastInto<(S1::Output, NewAxis)>,
    (S2, NewAxis): BroadcastInto<(S1::Output, NewAxis)>,
    S1: BroadcastInto<S1::Output>,
    S2: BroadcastInto<S1::Output>,
{
    type Output = (S1::Output, NewAxis);

    fn broadcast_together(self, other: (S2, NewAxis)) -> Option<Self::Output> {
        Some((self.0.broadcast_together(other.0)?, NewAxis))
    }
}

impl<S1: BroadcastTogether<S2>, S2: Shape, const N: usize> BroadcastTogether<(S2, Const<N>)>
    for (S1, NewAxis)
where
    (S1, NewAxis): BroadcastInto<(S1::Output, Const<N>)>,
    (S2, Const<N>): BroadcastInto<(S1::Output, Const<N>)>,
    (S1::Output, Const<N>): Shape,
    S2: BroadcastInto<S1::Output>,
{
    type Output = (S1::Output, Const<N>);

    fn broadcast_together(self, other: (S2, Const<N>)) -> Option<Self::Output> {
        Some((self.0.broadcast_together(other.0)?, Const))
    }
}

impl<S1: BroadcastTogether<S2>, S2: Shape> BroadcastTogether<(S2, usize)> for (S1, NewAxis)
where
    (S1, NewAxis): BroadcastInto<(S1::Output, usize)>,
    (S2, usize): BroadcastInto<(S1::Output, usize)>,
    (S1::Output, usize): Shape,
    S2: BroadcastInto<S1::Output>,
{
    type Output = (S1::Output, usize);

    fn broadcast_together(self, other: (S2, usize)) -> Option<Self::Output> {
        Some((self.0.broadcast_together(other.0)?, other.1))
    }
}

impl<S1: Shape, const N: usize> BroadcastTogether<()> for (S1, Const<N>)
where
    (S1, Const<N>): BroadcastInto<(S1, Const<N>)>,
    (): BroadcastInto<(S1, Const<N>)>,
{
    type Output = (S1, Const<N>);

    fn broadcast_together(self, _other: ()) -> Option<Self::Output> {
        Some(self)
    }
}

impl<S1: BroadcastTogether<S2>, S2: Shape, const N: usize> BroadcastTogether<(S2, NewAxis)>
    for (S1, Const<N>)
where
    (S1, Const<N>): BroadcastInto<(S1::Output, Const<N>)>,
    (S2, NewAxis): BroadcastInto<(S1::Output, Const<N>)>,
    (S1::Output, Const<N>): Shape,
    S2: BroadcastInto<S1::Output>,
{
    type Output = (S1::Output, Const<N>);

    fn broadcast_together(self, other: (S2, NewAxis)) -> Option<Self::Output> {
        Some((self.0.broadcast_together(other.0)?, Const))
    }
}

impl<S1: BroadcastTogether<S2>, S2: Shape, const N: usize> BroadcastTogether<(S2, Const<N>)>
    for (S1, Const<N>)
where
    (S1, Const<N>): BroadcastInto<(S1::Output, Const<N>)>,
    (S2, Const<N>): BroadcastInto<(S1::Output, Const<N>)>,
    (S1::Output, Const<N>): Shape,
    S2: BroadcastInto<S1::Output>,
{
    type Output = (S1::Output, Const<N>);

    fn broadcast_together(self, other: (S2, Const<N>)) -> Option<Self::Output> {
        Some((self.0.broadcast_together(other.0)?, Const))
    }
}

impl<S1: BroadcastTogether<S2>, S2: Shape, const N: usize> BroadcastTogether<(S2, usize)>
    for (S1, Const<N>)
where
    (S1, Const<N>): BroadcastInto<(S1::Output, Const<N>)>,
    (S2, usize): BroadcastInto<(S1::Output, Const<N>)>,
    (S1::Output, Const<N>): Shape,
    S2: BroadcastInto<S1::Output>,
{
    type Output = (S1::Output, Const<N>);

    fn broadcast_together(self, other: (S2, usize)) -> Option<Self::Output> {
        if N != other.1 {
            return None;
        }
        Some((self.0.broadcast_together(other.0)?, Const))
    }
}

impl<S1: Shape> BroadcastTogether<()> for (S1, usize)
where
    (S1, usize): BroadcastInto<(S1, usize)>,
    (): BroadcastInto<(S1, usize)>,
{
    type Output = (S1, usize);

    fn broadcast_together(self, _other: ()) -> Option<Self::Output> {
        Some(self)
    }
}

impl<S1: BroadcastTogether<S2>, S2: Shape> BroadcastTogether<(S2, NewAxis)> for (S1, usize)
where
    (S1, usize): BroadcastInto<(S1::Output, usize)>,
    (S2, NewAxis): BroadcastInto<(S1::Output, usize)>,
    (S1::Output, usize): Shape,
    S2: BroadcastInto<S1::Output>,
{
    type Output = (S1::Output, usize);

    fn broadcast_together(self, other: (S2, NewAxis)) -> Option<Self::Output> {
        Some((self.0.broadcast_together(other.0)?, self.1))
    }
}

impl<S1: BroadcastTogether<S2>, S2: Shape, const N: usize> BroadcastTogether<(S2, Const<N>)>
    for (S1, usize)
where
    (S1, usize): BroadcastInto<(S1::Output, Const<N>)>,
    (S2, Const<N>): BroadcastInto<(S1::Output, Const<N>)>,
    (S1::Output, Const<N>): Shape,
    S2: BroadcastInto<S1::Output>,
{
    type Output = (S1::Output, Const<N>);

    fn broadcast_together(self, other: (S2, Const<N>)) -> Option<Self::Output> {
        if self.1 != N {
            return None;
        }
        Some((self.0.broadcast_together(other.0)?, Const))
    }
}

impl<S1: BroadcastTogether<S2>, S2: Shape> BroadcastTogether<(S2, usize)> for (S1, usize)
where
    (S1, usize): BroadcastInto<(S1::Output, usize)>,
    (S2, usize): BroadcastInto<(S1::Output, usize)>,
    (S1::Output, usize): Shape,
    S2: BroadcastInto<S1::Output>,
{
    type Output = (S1::Output, usize);

    fn broadcast_together(self, other: (S2, usize)) -> Option<Self::Output> {
        if self.1 != other.1 {
            return None;
        }
        Some((self.0.broadcast_together(other.0)?, self.1))
    }
}

/*
// OLD STUFF //
// Base case-- 0D

impl AsIndex for () {
    type Index = [usize; 0];

    fn as_index(&self) -> Self::Index {
        []
    }
}

impl Shape for () {}

impl ShapeEq<()> for () {
    fn shape_eq(&self, _other: &()) -> bool {
        true
    }
}

impl BroadcastShape<()> for () {
    type Output = ();

    fn broadcast_shape(self, _other: ()) -> Option<()> {
        Some(())
    }
}
impl BroadcastShapeNoAlias<()> for () {}

impl IntoIndex<()> for () {
    fn into_index(_index: [usize; 0]) -> [usize; 0] {
        []
    }
}

// Recursive case

macro_rules! impl_shape {
    ($($An:ident)*, $($an:ident)*) => {

        // impl AsIndex for tuples that look like (Const<N>, ...)
        impl<const N: usize, $($An: Copy,)*> AsIndex for (Const<N>, $($An,)*)
        where
            ($($An,)*): Shape<Index: FixedLengthArray<OneBigger: Index>>,
        {
            type Index = <<($($An,)*) as AsIndex>::Index as FixedLengthArray>::OneBigger;

            fn as_index(&self) -> Self::Index {
                let &(_, $($an,)*) = self;
                ($($an,)*).as_index().prepend(N)
            }
        }
        impl<const N: usize, $($An: Copy + fmt::Debug,)*> Shape for (Const<N>, $($An,)*)
        where
            ($($An,)*): Shape<Index: FixedLengthArray<OneBigger: Index>>, {}

        // impl AsIndex for tuples that look like (usize, ...)
        impl<$($An: Copy,)*> AsIndex for (usize, $($An,)*)
        where
            ($($An,)*): Shape<Index: FixedLengthArray<OneBigger: Index>>,
        {
            type Index = <<($($An,)*) as AsIndex>::Index as FixedLengthArray>::OneBigger;

            fn as_index(&self) -> Self::Index {
                let &(e, $($an,)*) = self;
                ($($an,)*).as_index().prepend(e)
            }
        }
        impl<$($An: Copy + fmt::Debug,)*> Shape for (usize, $($An,)*)
        where
            ($($An,)*): Shape<Index: FixedLengthArray<OneBigger: Index>>,
        {}

        // impl AsIndex for tuples that look like (NewAxis, ...)
        impl<$($An: Copy,)*> AsIndex for (NewAxis, $($An,)*)
        where
            ($($An,)*): Shape,
        {
            type Index = <($($An,)*) as AsIndex>::Index;

            fn as_index(&self) -> Self::Index {
                let &(_, $($an,)*) = self;
                ($($an,)*).as_index()
            }
        }
        impl<$($An: Copy + fmt::Debug,)*> Shape for (NewAxis, $($An,)*)
        where
            ($($An,)*): Shape,
        {}
    };
}

macro_rules! impl_shape_eq {
    (,,,) => {
        // ShapeEq<(Const<N>, ...)> for (Const<N>, ...)
        impl<const N: usize> ShapeEq<(Const<N>,)> for (Const<N>,)
        where
            (Const<N>,): Shape,
        {
            fn shape_eq(&self, _other: &(Const<N>,)) -> bool {
                true
            }
        }

        // ShapeEq<(usize, ...)> for (Const<N>, ...)
        impl<const N: usize,  > ShapeEq<(usize, )> for (Const<N>, )
        where
            (Const<N>, ): Shape<Index = <(usize, ) as AsIndex>::Index>,
            (usize, ): Shape,
        {
            fn shape_eq(&self, other: &(usize, )) -> bool {
                let &(_, ) = self;
                let &(e, ) = other;
                N == e
            }
        }

        // ShapeEq<(Const<N>, ...)> for (usize, ...)
        impl<const N: usize,  > ShapeEq<(Const<N>, )> for (usize, )
        where
            (usize, ): Shape<Index = <(Const<N>, ) as AsIndex>::Index>,
            (Const<N>, ): Shape,
        {
            fn shape_eq(&self, other: &(Const<N>, )) -> bool {
                let &(e, ) = self;
                let &(_, ) = other;
                e == N
            }
        }

        // ShapeEq<(usize, ...)> for (usize, ...)
        impl ShapeEq<(usize, )> for (usize, )
        where
            (usize, ): Shape,
        {
            fn shape_eq(&self, other: &(usize, )) -> bool {
                let &(e, ) = self;
                let &(f, ) = other;
                e == f
            }
        }

        // ShapeEq<(NewAxis, ...)> for (NewAxis, ...)
        impl< > ShapeEq<(NewAxis, )> for (NewAxis, )
        where
            (NewAxis, ): Shape,
        {
            fn shape_eq(&self, _other: &(NewAxis, )) -> bool {
                true
            }
        }
    };
    ($($An:ident)*, $($an:ident)*, $($Bn:ident)*, $($bn:ident)*) => {

        // ShapeEq<(Const<N>, ...)> for (Const<N>, ...)
        impl<const N: usize, $($An: Copy,)* $($Bn: Copy,)*> ShapeEq<(Const<N>, $($Bn,)*)> for (Const<N>, $($An,)*)
        where
            (Const<N>, $($An,)*): Shape<Index = <(Const<N>, $($Bn,)*) as AsIndex>::Index>,
            (Const<N>, $($Bn,)*): Shape,
            ($($An,)*): ShapeEq<($($Bn,)*)>,
            ($($Bn,)*): Shape,
        {
            fn shape_eq(&self, other: &(Const<N>, $($Bn,)*)) -> bool {
                let &(_, $($an,)*) = self;
                let &(_, $($bn,)*) = other;
                ($($an,)*).shape_eq(&($($bn,)*))
            }
        }

        // ShapeEq<(usize, ...)> for (Const<N>, ...)
        impl<const N: usize, $($An: Copy,)* $($Bn: Copy,)*> ShapeEq<(usize, $($Bn,)*)> for (Const<N>, $($An,)*)
        where
            (Const<N>, $($An,)*): Shape<Index = <(usize, $($Bn,)*) as AsIndex>::Index>,
            (usize, $($Bn,)*): Shape,
            ($($An,)*): ShapeEq<($($Bn,)*)>,
            ($($Bn,)*): Shape,
        {
            fn shape_eq(&self, other: &(usize, $($Bn,)*)) -> bool {
                let &(_, $($an,)*) = self;
                let &(e, $($bn,)*) = other;
                N == e && ($($an,)*).shape_eq(&($($bn,)*))
            }
        }


        // ShapeEq<(Const<N>, ...)> for (usize, ...)
        impl<const N: usize, $($An: Copy,)* $($Bn: Copy,)*> ShapeEq<(Const<N>, $($Bn,)*)> for (usize, $($An,)*)
        where
            (usize, $($An,)*): Shape<Index = <(Const<N>, $($Bn,)*) as AsIndex>::Index>,
            (Const<N>, $($Bn,)*): Shape,
            ($($An,)*): ShapeEq<($($Bn,)*)>,
            ($($Bn,)*): Shape,
        {
            fn shape_eq(&self, other: &(Const<N>, $($Bn,)*)) -> bool {
                let &(e, $($an,)*) = self;
                let &(_, $($bn,)*) = other;
                e == N && ($($an,)*).shape_eq(&($($bn,)*))
            }
        }

        // ShapeEq<(usize, ...)> for (usize, ...)
        impl<$($An: Copy,)* $($Bn: Copy,)*> ShapeEq<(usize, $($Bn,)*)> for (usize, $($An,)*)
        where
            (usize, $($An,)*): Shape<Index = <(usize, $($Bn,)*) as AsIndex>::Index>,
            (usize, $($Bn,)*): Shape,
            ($($An,)*): ShapeEq<($($Bn,)*)>,
            ($($Bn,)*): Shape,
        {
            fn shape_eq(&self, other: &(usize, $($Bn,)*)) -> bool {
                let &(e, $($an,)*) = self;
                let &(f, $($bn,)*) = other;
                e == f && ($($an,)*).shape_eq(&($($bn,)*))
            }
        }

        // ShapeEq<(NewAxis, ...)> for (NewAxis, ...)
        impl<$($An: Copy,)* $($Bn: Copy,)*> ShapeEq<(NewAxis, $($Bn,)*)> for (NewAxis, $($An,)*)
        where
            (NewAxis, $($An,)*): Shape<Index = <(NewAxis, $($Bn,)*) as AsIndex>::Index>,
            (NewAxis, $($Bn,)*): Shape,
            ($($An,)*): ShapeEq<($($Bn,)*)>,
            ($($Bn,)*): Shape,
        {
            fn shape_eq(&self, other: &(NewAxis, $($Bn,)*)) -> bool {
                let &(_, $($an,)*) = self;
                let &(_, $($bn,)*) = other;
                ($($an,)*).shape_eq(&($($bn,)*))
            }
        }
    };
}

macro_rules! impl_shape_eq_reduction {
    (,,,, $($NewAxis:ident)*, $($NAUnderscore:ident)*) => {
        // ShapeEq<(...)> for (NewAxis, ...)
        // ShapeEq<(...)> for (NewAxis, NewAxis, ...)
        // etc.
        impl ShapeEq<()> for ($($NewAxis,)*)
        {
            fn shape_eq(&self, _other: &()) -> bool {
                true
            }
        }

        // ShapeEq<(NewAxis, ...)> for (...)
        // ShapeEq<(NewAxis, NewAxis, ...)> for (...)
        // etc.
        impl ShapeEq<($($NewAxis,)*)> for ()
        {
            fn shape_eq(&self, _other: &($($NewAxis,)*)) -> bool {
                true
            }
        }
    };
    ($($An:ident)*, $($an:ident)*, $($Bn:ident)*, $($bn:ident)*, $($NewAxis:ident)*, $($NAUnderscore:ident)*) => {
        // ShapeEq<(...)> for (NewAxis, ...)
        // ShapeEq<(...)> for (NewAxis, NewAxis, ...)
        // etc.
        impl<$($An: Copy,)* $($Bn: Copy,)*> ShapeEq<($($Bn,)*)> for ($($NewAxis,)* $($An,)*)
        where
            ($($NewAxis,)* $($An,)*): Shape<Index = <($($Bn,)*) as AsIndex>::Index>,
            ($($Bn,)*): Shape,
            ($($An,)*): ShapeEq<($($Bn,)*)>,
        {
            fn shape_eq(&self, other: &($($Bn,)*)) -> bool {
                let &($($NAUnderscore,)* $($an,)*) = self;
                let &($($bn,)*) = other;
                ($($an,)*).shape_eq(&($($bn,)*))
            }
        }

        // ShapeEq<(NewAxis, ...)> for (...)
        // ShapeEq<(NewAxis, NewAxis, ...)> for (...)
        // etc.
        impl<$($An: Copy,)* $($Bn: Copy,)*> ShapeEq<($($NewAxis,)* $($Bn,)*)> for ($($An,)*)
        where
            ($($An,)*): Shape<Index = <($($NewAxis,)* $($Bn,)*) as AsIndex>::Index>,
            ($($NewAxis,)* $($Bn,)*): Shape,
            ($($Bn,)*): Shape,
            ($($An,)*): ShapeEq<($($Bn,)*)>,
        {
            fn shape_eq(&self, other: &($($NewAxis,)* $($Bn,)*)) -> bool {
                let &($($an,)*) = self;
                let &($($NAUnderscore,)* $($bn,)*) = other;
                ($($an,)*).shape_eq(&($($bn,)*))
            }
        }
    };
}

macro_rules! impl_into_index {
    (,,,) => {
        // IntoIndex<(Const<N>, ...)> for (Const<N>, ...)
        impl<const N: usize> IntoIndex<(Const<N>,)> for (Const<N>,)
        where
            (Const<N>,): Shape,
        {
            fn into_index(index: Self::Index) -> Self::Index {
                index
            }
        }

        // IntoIndex<(usize, ...)> for (Const<N>, ...)
        impl<const N: usize> IntoIndex<(usize, )> for (Const<N>,)
        {
            fn into_index(index: Self::Index) -> Self::Index {
                index
            }
        }

        // IntoIndex<(Const<N>, ...)> for (usize, ...)
        impl<const N: usize> IntoIndex<(Const<N>,)> for (usize,)
        {
            fn into_index(index: Self::Index) -> Self::Index {
                index
            }
        }

        // IntoIndex<(usize, ...)> for (usize, ...)
        impl IntoIndex<(usize,)> for (usize,)
        where
            (usize,): Shape,
        {
            fn into_index(index: Self::Index) -> Self::Index {
                index
            }
        }
    };

    ($($An:ident)*, $($an:ident)*, $($Bn:ident)*, $($bn:ident)*) => {
        // IntoIndex<(Const<N>, ...)> for (Const<N>, ...)
        impl<const N: usize, $($An: Copy,)* $($Bn: Copy,)*> IntoIndex<(Const<N>, $($Bn,)*)> for (Const<N>, $($An,)*)
        where
            ($($An,)*): IntoIndex<($($Bn,)*)>,
            <($($An,)*) as AsIndex>::Index: FixedLengthArray<OneBigger=Self::Index>,
        {
            fn into_index(index: Self::Index) -> Self::Index {
                index
            }
        }

        // IntoIndex<(usize, ...)> for (Const<N>, ...)
        impl<const N: usize, $($An: Copy,)* $($Bn: Copy,)*> IntoIndex<(usize, $($Bn,)*)> for (Const<N>, $($An,)*)
        where
            (Const<N>, $($An,)*): Shape<Index = <(usize, $($Bn,)*) as AsIndex>::Index>,
            (usize, $($Bn,)*): Shape,
        {
            fn into_index(index: Self::Index) -> Self::Index {
                index
            }
        }

        // IntoIndex<(Const<N>, ...)> for (usize, ...)
        impl<const N: usize, $($An: Copy,)* $($Bn: Copy,)*> IntoIndex<(Const<N>, $($Bn,)*)> for (usize, $($An,)*)
        where
            (usize, $($An,)*): Shape<Index = <(Const<N>, $($Bn,)*) as AsIndex>::Index>,
            (Const<N>, $($Bn,)*): Shape,
        {
            fn into_index(index: Self::Index) -> Self::Index {
                index
            }
        }

        // IntoIndex<(usize, ...)> for (usize, ...)
        impl<$($An: Copy,)* $($Bn: Copy,)*> IntoIndex<(usize, $($Bn,)*)> for (usize, $($An,)*)
        where
            (usize, $($An,)*): Shape<Index = <(usize, $($Bn,)*) as AsIndex>::Index>,
            (usize, $($Bn,)*): Shape,
        {
            fn into_index(index: Self::Index) -> Self::Index {
                index
            }
        }

        // IntoIndex<(Const<N>, ...)> for (NewAxis, ...)
        impl<const N: NewAxis, $($An: Copy,)* $($Bn: Copy,)*> IntoIndex<(Const<N>, $($Bn,)*)> for (NewAxis, $($An,)*)
        where
            //(NewAxis, $($An,)*): Shape<Index = <(Const<N>, $($Bn,)*) as AsIndex>::Index>,
            //(Const<N>, $($Bn,)*): Shape,
        {
            fn into_index(index: Self::Index) -> Self::Index::OneBigger {
                index
            }
        }

    };

}

macro_rules! impl_broadcast {
    (,,,,,) => {

        // BroadcastShape<(Const<N>, ...)> for (Const<N>, ...)
        impl<const N: usize> BroadcastShape<(Const<N>,)> for (Const<N>,)
        {
            type Output = (Const<N>,);
            fn broadcast_shape(self, _other: (Const<N>,)) -> Option<Self::Output> {
                Some((Const, ))
            }
        }
    };
    ($($An:ident)*, $($an:ident)*, $($Bn:ident)*, $($bn:ident)*, $($Cn:ident)*, $($cn:ident)*) => {

        // BroadcastShape<(Const<N>, ...)> for (Const<N>, ...)
        impl<const N: usize, $($An: Copy,)* $($Bn: Copy,)* $($Cn: Copy,)*> BroadcastShape<(Const<N>, $($Bn,)*)> for (Const<N>, $($An,)*)
        where
            ($($An,)*): BroadcastShape<($($Bn,)*), Output = ($($Cn,)*)>,
            ($($Bn,)*): Shape + IntoIndex<($($Cn,)*)>,
            ($($Cn,)*): Shape,
            (Const<N>, $($Cn,)*): Shape,
            (Const<N>, $($An,)*): IntoIndex<(Const<N>, $($Cn,)*)>,
            (Const<N>, $($Bn,)*): IntoIndex<(Const<N>, $($Cn,)*)>,
        {
            type Output = (Const<N>, $($Cn,)*);
            fn broadcast_shape(self, other: (Const<N>, $($Bn,)*)) -> Option<Self::Output> {
                let (_, $($an,)*) = self;
                let (_, $($bn,)*) = other;
                let ($($cn,)*) = ($($an,)*).broadcast_shape(($($bn,)*))?;
                Some((Const, $($cn,)*))
            }
        }

/*
        // BroadcastShape<(usize, ...)> for (Const<N>, ...)
        impl<const N: usize, $($An: Copy,)* $($Bn: Copy,)*> BroadcastShape<(usize, $($Bn,)*)> for (Const<N>, $($An,)*)
        where
            (Const<N>, $($An,)*): Shape<Index = <(usize, $($Bn,)*) as AsIndex>::Index>,
            (usize, $($Bn,)*): Shape,
            ($($An,)*): BroadcastShape<($($Bn,)*)>,
            ($($Bn,)*): Shape,
        {
            fn broadcast_shape(&self, other: &(usize, $($Bn,)*)) -> bool {
                let &(_, $($an,)*) = self;
                let &(e, $($bn,)*) = other;
                N == e && ($($an,)*).broadcast_shape(&($($bn,)*))
            }
        }

        // BroadcastShape<(NewAxis, ...)> for (Const<N>, ...)
        impl<const N: NewAxis, $($An: Copy,)* $($Bn: Copy,)*> BroadcastShape<(NewAxis, $($Bn,)*)> for (Const<N>, $($An,)*)
        where
            (Const<N>, $($An,)*): Shape<Index = <(NewAxis, $($Bn,)*) as AsIndex>::Index>,
            (NewAxis, $($Bn,)*): Shape,
            ($($An,)*): BroadcastShape<($($Bn,)*)>,
            ($($Bn,)*): Shape,
        {
            fn broadcast_shape(&self, other: &(NewAxis, $($Bn,)*)) -> bool {
                let &(_, $($an,)*) = self;
                let &(_, $($bn,)*) = other;
                ($($an,)*).broadcast_shape(&($($bn,)*))
            }
        }

        // BroadcastShape<(Const<N>, ...)> for (usize, ...)
        impl<const N: usize, $($An: Copy,)* $($Bn: Copy,)*> BroadcastShape<(Const<N>, $($Bn,)*)> for (usize, $($An,)*)
        where
            (usize, $($An,)*): Shape<Index = <(Const<N>, $($Bn,)*) as AsIndex>::Index>,
            (Const<N>, $($Bn,)*): Shape,
            ($($An,)*): BroadcastShape<($($Bn,)*)>,
            ($($Bn,)*): Shape,
        {
            fn broadcast_shape(&self, other: &(Const<N>, $($Bn,)*)) -> bool {
                let &(e, $($an,)*) = self;
                let &(_, $($bn,)*) = other;
                e == N && ($($an,)*).broadcast_shape(&($($bn,)*))
            }
        }

        // BroadcastShape<(usize, ...)> for (usize, ...)
        impl<$($An: Copy,)* $($Bn: Copy,)*> BroadcastShape<(usize, $($Bn,)*)> for (usize, $($An,)*)
        where
            (usize, $($An,)*): Shape<Index = <(usize, $($Bn,)*) as AsIndex>::Index>,
            (usize, $($Bn,)*): Shape,
            ($($An,)*): BroadcastShape<($($Bn,)*)>,
            ($($Bn,)*): Shape,
        {
            fn broadcast_shape(&self, other: &(usize, $($Bn,)*)) -> bool {
                let &(e, $($an,)*) = self;
                let &(f, $($bn,)*) = other;
                e == f && ($($an,)*).broadcast_shape(&($($bn,)*))
            }
        }

        // BroadcastShape<(NewAxis, ...)> for (usize, ...)
        impl<const N: NewAxis, $($An: Copy,)* $($Bn: Copy,)*> BroadcastShape<(NewAxis, $($Bn,)*)> for (usize, $($An,)*)
        where
            (usize, $($An,)*): Shape<Index = <(NewAxis, $($Bn,)*) as AsIndex>::Index>,
            (NewAxis, $($Bn,)*): Shape,
            ($($An,)*): BroadcastShape<($($Bn,)*)>,
            ($($Bn,)*): Shape,
        {
            fn broadcast_shape(&self, other: &(NewAxis, $($Bn,)*)) -> bool {
                let &(e, $($an,)*) = self;
                let &(_, $($bn,)*) = other;
                ($($an,)*).broadcast_shape(&($($bn,)*))
            }
        }

        // BroadcastShape<(NewAxis, ...)> for (NewAxis, ...)
        impl<$($An: Copy,)* $($Bn: Copy,)*> BroadcastShape<(NewAxis, $($Bn,)*)> for (NewAxis, $($An,)*)
        where
            (NewAxis, $($An,)*): Shape<Index = <(NewAxis, $($Bn,)*) as AsIndex>::Index>,
            (NewAxis, $($Bn,)*): Shape,
            ($($An,)*): BroadcastShape<($($Bn,)*)>,
            ($($Bn,)*): Shape,
        {
            fn broadcast_shape(&self, other: &(NewAxis, $($Bn,)*)) -> bool {
                let &(_, $($an,)*) = self;
                let &(_, $($bn,)*) = other;
                ($($an,)*).broadcast_shape(&($($bn,)*))
            }
        }
*/

    };
}

impl_shape!(,); // 1D
impl_shape!(A0, a0); // 2D
impl_shape!(A0 A1, a0 a1); // 3D
impl_shape!(A0 A1 A2, a0 a1 a2); // 4D

impl_shape_eq!(,,,); // 1D
impl_shape_eq!(A0, a0, B0, b0); // 2D
impl_shape_eq!(A0 A1, a0 a1, B0 B1, b0 b1); // 3D
impl_shape_eq!(A0 A1 A2, a0 a1 a2, B0 B1 B2, b0 b1 b2); // 4D

impl_shape_eq_reduction!(,,,, NewAxis, _0); // 1D

impl_shape_eq_reduction!(,,,, NewAxis NewAxis, _0 _1); // 2D
impl_shape_eq_reduction!(A0, a0, B0, b0, NewAxis, _0); // 2D

impl_shape_eq_reduction!(,,,, NewAxis NewAxis NewAxis, _0 _1 _2); // 3D
impl_shape_eq_reduction!(A0, a0, B0, b0, NewAxis NewAxis, _0 _1); // 3D
impl_shape_eq_reduction!(A0 A1, a0 a1, B0 B1, b0 b1, NewAxis, _0); // 3D

impl_shape_eq_reduction!(,,,, NewAxis NewAxis NewAxis NewAxis, _0 _1 _2 _3); // 4D
impl_shape_eq_reduction!(A0, a0, B0, b0, NewAxis NewAxis NewAxis, _0 _1 _2); // 4D
impl_shape_eq_reduction!(A0 A1, a0 a1, B0 B1, b0 b1, NewAxis NewAxis, _0 _2); // 4D
impl_shape_eq_reduction!(A0 A1 A2, a0 a1 a2, B0 B1 B2, b0 b1 b2, NewAxis, _0); // 4D

impl_into_index!(,,,); // 1D
impl_into_index!(A0, a0, B0, b0); // 2D
impl_into_index!(A0 A1, a0 a1, B0 B1, b0 b1); // 3D
impl_into_index!(A0 A1 A2, a0 a1 a2, B0 B1 B2, b0 b1 b2); // 4D

impl_broadcast!(,,,,,); // 1D
impl_broadcast!(A0, a0, B0, b0, C0, c0); // 3D
impl_broadcast!(A0 A1, a0 a1, B0 B1, b0 b1, C0 C1, c0 c1); // 3D
impl_broadcast!(A0 A1 A2, a0 a1 a2, B0 B1 B2, b0 b1 b2, C0 C1 C2, c0 c1 c2); // 4D

*/

/*
impl<D1: BroadcastDim<NewAxis, Output: Dim>> BroadcastShape<()> for (D1,) {
    type Output = (D1::Output,);

    fn broadcast_shape(self, _other: ()) -> Option<Self::Output> {
        Some((self.0.broadcast(NewAxis)?,))
    }
}
impl<D1: BroadcastNoAlias<NewAxis>> BroadcastNoAlias<()> for (D1,) {}

/////

impl<D1: BroadcastDim<NewAxis, Output: Dim>> BroadcastShape<()> for (D1,) {
    type Output = (D1::Output,);

    fn broadcast_shape(self, _other: ()) -> Option<Self::Output> {
        Some((self.0.broadcast(NewAxis)?,))
    }
}
impl<D1: BroadcastNoAlias<NewAxis>> BroadcastNoAlias<()> for (D1,) {}

impl<E1> BroadcastShape<(E1,)> for ()
where
    NewAxis: BroadcastDim<E1>,
{
    type Output = (<NewAxis as BroadcastDim<E1>>::Output,);

    fn broadcast_shape(self, other: (E1,)) -> Option<Self::Output> {
        Some((NewAxis.broadcast(other.0)?,))
    }
}
impl<E1> BroadcastNoAlias<(E1,)> for () where NewAxis: BroadcastNoAlias<E1> {}

impl<D1: BroadcastDim<E1>, E1> BroadcastShape<(E1,)> for (D1,) {
    type Output = (<D1 as BroadcastDim<E1>>::Output,);
    fn broadcast_shape(self, other: (E1,)) -> Option<Self::Output> {
        Some((self.0.broadcast(other.0)?,))
    }
}
impl<D1: BroadcastDimNoAlias<E1>, E1> BroadcastShapeNoAlias<(E1,)> for (D1,) {}

impl<E1> ConvertIndex<(E1,)> for ()
where
    (E1,): Shape<Index = [usize; 1]>,
    NewAxis: BroadcastDim<E1>,
{
    fn convert_index(_index: [usize; 0]) -> [usize; 1] {
        [0]
    }
}

impl<D1: BroadcastDim<E1>, E1> ConvertIndex<(E1,)> for (D1,) {
    fn convert_index(index: [usize; 1]) -> [usize; 1] {
        index
    }
}

// 2 axes

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

impl<D1: Dim> AsIndex for (D1, NewAxis) {
    type Index = [usize; 1];

    fn as_index(&self) -> Self::Index {
        [self.0.into()]
    }
}

impl<D1: Dim> Shape for (D1, NewAxis) {}

impl<D1: Dim + PartialEq<E1>, E1: Dim> ShapeEq<(E1, NewAxis)> for (D1, NewAxis) {
    fn shape_eq(&self, other: &(E1, NewAxis)) -> bool {
        self.0 == other.0
    }
}

impl<D1: BroadcastDim<NewAxis>, D2: BroadcastDim<NewAxis>> BroadcastShape<()> for (D1, D2) {
    type Output = (D1::Output, D2::Output);

    fn broadcast_shape(self, _other: ()) -> Option<Self::Output> {
        Some((self.0.broadcast(NewAxis)?, self.1.broadcast(NewAxis)?))
    }
}
impl<D1: BroadcastNoAlias<NewAxis>, D2: BroadcastNoAlias<NewAxis>> BroadcastNoAlias<()>
    for (D1, D2)
{
}

impl<E1, E2> BroadcastShape<(E1, E2)> for ()
where
    NewAxis: BroadcastDim<E1>,
    NewAxis: BroadcastDim<E2>,
{
    type Output = (
        <NewAxis as BroadcastDim<E1>>::Output,
        <NewAxis as BroadcastDim<E2>>::Output,
    );

    fn broadcast_shape(self, other: (E1, E2)) -> Option<Self::Output> {
        Some((NewAxis.broadcast(other.0)?, NewAxis.broadcast(other.1)?))
    }
}
impl<E1, E2> BroadcastNoAlias<(E1, E2)> for ()
where
    NewAxis: BroadcastNoAlias<E1>,
    NewAxis: BroadcastNoAlias<E2>,
{
}

impl<D1: BroadcastDim<NewAxis>, D2: BroadcastDim<E1>, E1> BroadcastShape<(E1,)> for (D1, D2) {
    type Output = (D1::Output, D2::Output);

    fn broadcast_shape(self, other: (E1,)) -> Option<Self::Output> {
        Some((self.0.broadcast(NewAxis)?, self.1.broadcast(other.0)?))
    }
}
impl<D1: BroadcastNoAlias<NewAxis>, D2: BroadcastNoAlias<E1>, E1> BroadcastNoAlias<(E1,)>
    for (D1, D2)
{
}

impl<D1: BroadcastDim<E2>, E1, E2> BroadcastShape<(E1, E2)> for (D1,)
where
    NewAxis: BroadcastDim<E1>,
{
    type Output = (<NewAxis as BroadcastDim<E1>>::Output, D1::Output);

    fn broadcast_shape(self, other: (E1, E2)) -> Option<Self::Output> {
        Some((NewAxis.broadcast(other.0)?, self.0.broadcast(other.1)?))
    }
}
impl<D1: BroadcastNoAlias<E2>, E1, E2> BroadcastNoAlias<(E1, E2)> for (D1,) where
    NewAxis: BroadcastNoAlias<E1>
{
}

impl<D1: BroadcastDim<E1>, D2: BroadcastDim<E2>, E1, E2> BroadcastShape<(E1, E2)> for (D1, D2) {
    type Output = (D1::Output, D2::Output);

    fn broadcast_shape(self, other: (E1, E2)) -> Option<Self::Output> {
        Some((self.0.broadcast(other.0)?, self.1.broadcast(other.1)?))
    }
}
impl<D1: BroadcastNoAlias<E1>, D2: BroadcastNoAlias<E2>, E1, E2> BroadcastNoAlias<(E1, E2)>
    for (D1, D2)
{
}

// 3 axes

impl<D1: Dim, D2: Dim, D3: Dim> AsIndex for (D1, D2, D3) {
    type Index = [usize; 3];

    fn as_index(&self) -> Self::Index {
        [self.0.into(), self.1.into(), self.2.into()]
    }
}

impl<D1: Dim, D2: Dim, D3: Dim> Shape for (D1, D2, D3) {}

impl<
        D1: Dim + PartialEq<E1>,
        D2: Dim + PartialEq<E2>,
        D3: Dim + PartialEq<E3>,
        E1: Dim,
        E2: Dim,
        E3: Dim,
    > ShapeEq<(E1, E2, E3)> for (D1, D2, D3)
{
    fn shape_eq(&self, other: &(E1, E2, E3)) -> bool {
        self.0 == other.0 && self.1 == other.1 && self.2 == other.2
    }
}

impl<D1: Dim, D2: Dim> AsIndex for (D1, D2, NewAxis) {
    type Index = [usize; 2];

    fn as_index(&self) -> Self::Index {
        [self.0.into(), self.1.into()]
    }
}

impl<D1: Dim, D2: Dim> Shape for (D1, D2, NewAxis) {}

impl<D1: Dim + PartialEq<E1>, D2: Dim + PartialEq<E2>, E1: Dim, E2: Dim> ShapeEq<(E1, E2, NewAxis)>
    for (D1, D2, NewAxis)
{
    fn shape_eq(&self, other: &(E1, E2, NewAxis)) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<D1: Dim, D2: Dim> AsIndex for (D1, NewAxis, D2) {
    type Index = [usize; 2];

    fn as_index(&self) -> Self::Index {
        [self.0.into(), self.2.into()]
    }
}

impl<D1: Dim, D2: Dim> Shape for (D1, NewAxis, D2) {}

impl<D1: Dim + PartialEq<E1>, D2: Dim + PartialEq<E2>, E1: Dim, E2: Dim> ShapeEq<(E1, NewAxis, E2)>
    for (D1, NewAxis, D2)
{
    fn shape_eq(&self, other: &(E1, NewAxis, E2)) -> bool {
        self.0 == other.0 && self.2 == other.2
    }
}

impl<D1: Dim> AsIndex for (D1, NewAxis, NewAxis) {
    type Index = [usize; 1];

    fn as_index(&self) -> Self::Index {
        [self.0.into()]
    }
}

impl<D1: Dim> Shape for (D1, NewAxis, NewAxis) {}

impl<D1: Dim + cmp::PartialEq<E1>, E1: Dim> ShapeEq<(E1, NewAxis, NewAxis)>
    for (D1, NewAxis, NewAxis)
{
    fn shape_eq(&self, other: &(E1, NewAxis, NewAxis)) -> bool {
        self.0 == other.0
    }
}

impl<D1: BroadcastDim<NewAxis>, D2: BroadcastDim<NewAxis>, D3: BroadcastDim<NewAxis>>
    BroadcastShape<()> for (D1, D2, D3)
{
    type Output = (D1::Output, D2::Output, D3::Output);

    fn broadcast_shape(self, _other: ()) -> Option<Self::Output> {
        Some((
            self.0.broadcast(NewAxis)?,
            self.1.broadcast(NewAxis)?,
            self.2.broadcast(NewAxis)?,
        ))
    }
}
impl<
        D1: BroadcastDimNoAlias<NewAxis>,
        D2: BroadcastDimNoAlias<NewAxis>,
        D3: BroadcastDimNoAlias<NewAxis>,
    > BroadcastShapeNoAlias<()> for (D1, D2, D3)
{
}

impl<E1, E2, E3> BroadcastShape<(E1, E2, E3)> for ()
where
    NewAxis: BroadcastDim<E1>,
    NewAxis: BroadcastDim<E2>,
    NewAxis: BroadcastDim<E3>,
{
    type Output = (
        <NewAxis as BroadcastDim<E1>>::Output,
        <NewAxis as BroadcastDim<E2>>::Output,
        <NewAxis as BroadcastDim<E3>>::Output,
    );

    fn broadcast_shape(self, other: (E1, E2, E3)) -> Option<Self::Output> {
        Some((
            NewAxis.broadcast(other.0)?,
            NewAxis.broadcast(other.1)?,
            NewAxis.broadcast(other.2)?,
        ))
    }
}

impl<E1, E2, E3> BroadcastNoAlias<(E1, E2, E3)> for ()
where
    NewAxis: BroadcastNoAlias<E1>,
    NewAxis: BroadcastNoAlias<E2>,
    NewAxis: BroadcastNoAlias<E3>,
{
}

impl<D1: BroadcastDim<NewAxis>, D2: BroadcastDim<NewAxis>, D3: BroadcastDim<E1>, E1>
    BroadcastShape<(E1,)> for (D1, D2, D3)
{
    type Output = (D1::Output, D2::Output, D3::Output);

    fn broadcast_shape(self, other: (E1,)) -> Option<Self::Output> {
        Some((
            self.0.broadcast(NewAxis)?,
            self.1.broadcast(NewAxis)?,
            self.2.broadcast(other.0)?,
        ))
    }
}
impl<
        D1: BroadcastNoAlias<NewAxis>,
        D2: BroadcastNoAlias<NewAxis>,
        D3: BroadcastNoAlias<E1>,
        E1,
    > BroadcastNoAlias<(E1,)> for (D1, D2, D3)
{
}

impl<D1: BroadcastDim<E3>, E1, E2, E3> BroadcastShape<(E1, E2, E3)> for (D1,)
where
    NewAxis: BroadcastDim<E1>,
    NewAxis: BroadcastDim<E2>,
{
    type Output = (
        <NewAxis as BroadcastDim<E1>>::Output,
        <NewAxis as BroadcastDim<E2>>::Output,
        D1::Output,
    );

    fn broadcast_shape(self, other: (E1, E2, E3)) -> Option<Self::Output> {
        Some((
            NewAxis.broadcast(other.0)?,
            NewAxis.broadcast(other.1)?,
            self.0.broadcast(other.2)?,
        ))
    }
}
impl<D1: BroadcastNoAlias<E3>, E1, E2, E3> BroadcastNoAlias<(E1, E2, E3)> for (D1,)
where
    NewAxis: BroadcastNoAlias<E1>,
    NewAxis: BroadcastNoAlias<E2>,
{
}

impl<D1: BroadcastDim<NewAxis>, D2: BroadcastDim<E1>, D3: BroadcastDim<E2>, E1, E2>
    BroadcastShape<(E1, E2)> for (D1, D2, D3)
{
    type Output = (D1::Output, D2::Output, D3::Output);

    fn broadcast_shape(self, other: (E1, E2)) -> Option<Self::Output> {
        Some((
            self.0.broadcast(NewAxis)?,
            self.1.broadcast(other.0)?,
            self.2.broadcast(other.1)?,
        ))
    }
}
impl<D1: BroadcastNoAlias<NewAxis>, D2: BroadcastNoAlias<E1>, D3: BroadcastNoAlias<E2>, E1, E2>
    BroadcastNoAlias<(E1, E2)> for (D1, D2, D3)
{
}

impl<D1: BroadcastDim<E2>, D2: BroadcastDim<E3>, E1, E2, E3> BroadcastShape<(E1, E2, E3)>
    for (D1, D2)
where
    NewAxis: BroadcastDim<E1>,
{
    type Output = (
        <NewAxis as BroadcastDim<E1>>::Output,
        D1::Output,
        D2::Output,
    );

    fn broadcast_shape(self, other: (E1, E2, E3)) -> Option<Self::Output> {
        Some((
            NewAxis.broadcast(other.0)?,
            self.0.broadcast(other.1)?,
            self.1.broadcast(other.2)?,
        ))
    }
}
impl<D1: BroadcastNoAlias<E2>, D2: BroadcastNoAlias<E3>, E1, E2, E3> BroadcastNoAlias<(E1, E2, E3)>
    for (D1, D2)
where
    NewAxis: BroadcastNoAlias<E1>,
{
}

impl<D1: BroadcastDim<E1>, D2: BroadcastDim<E2>, D3: BroadcastDim<E3>, E1, E2, E3>
    BroadcastShape<(E1, E2, E3)> for (D1, D2, D3)
{
    type Output = (D1::Output, D2::Output, D3::Output);

    fn broadcast_shape(self, other: (E1, E2, E3)) -> Option<Self::Output> {
        Some((
            self.0.broadcast(other.0)?,
            self.1.broadcast(other.1)?,
            self.2.broadcast(other.2)?,
        ))
    }
}
impl<D1: BroadcastNoAlias<E1>, D2: BroadcastNoAlias<E2>, D3: BroadcastNoAlias<E3>, E1, E2, E3>
    BroadcastNoAlias<(E1, E2, E3)> for (D1, D2, D3)
{
}

*/

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

impl<S: Shape, S2: Shape + BroadcastInto<S>, D: ?Sized, D2: ops::Deref<Target = D>>
    IntoViewWithShape<S, D> for Array<S2, D2>
{
    fn view_with_shape(&self, shape: S) -> View<S, D> {
        self.shape.broadcast_into_fail(shape);
        View {
            shape,
            offset: self.offset,
            strides: S2::into_index(self.strides),
            data: &self.data,
        }
    }
}

impl<S: Shape + BroadcastInto<S>, D: ?Sized, D2: ops::Deref<Target = D>> IntoView<D>
    for Array<S, D2>
{
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

impl<S: Shape, S2: Shape + BroadcastInto<S>, D: ?Sized> IntoViewWithShape<S, D>
    for View<'_, S2, D>
{
    fn view_with_shape(&self, shape: S) -> View<S, D> {
        self.shape.broadcast_into_fail(shape);
        View {
            shape,
            offset: self.offset,
            strides: S2::into_index(self.strides),
            data: self.data,
        }
    }
}

impl<S: Shape + BroadcastInto<S>, D: ?Sized> IntoView<D> for View<'_, S, D> {
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

impl<S: Shape, S2: Shape + BroadcastInto<S>, D: ?Sized> IntoViewWithShape<S, D>
    for ViewMut<'_, S2, D>
{
    fn view_with_shape(&self, shape: S) -> View<S, D> {
        self.shape.broadcast_into_fail(shape);
        View {
            shape,
            offset: self.offset,
            strides: S2::into_index(self.strides),
            data: self.data,
        }
    }
}

impl<S: Shape + BroadcastInto<S>, D: ?Sized> IntoView<D> for ViewMut<'_, S, D> {
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
        S2: Shape + BroadcastIntoNoAlias<S>,
        D: ?Sized,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > IntoViewMutWithShape<S, D> for Array<S2, D2>
{
    fn view_mut_with_shape(&mut self, shape: S) -> ViewMut<'_, S, D> {
        self.shape.can_broadcast_into(shape);
        ViewMut {
            shape,
            offset: self.offset,
            strides: S2::into_index(self.strides),
            data: &mut self.data,
        }
    }
}

impl<
        S: Shape + BroadcastIntoNoAlias<S>,
        D: ?Sized,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > IntoViewMut<D> for Array<S, D2>
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

impl<S: Shape, S2: Shape + BroadcastIntoNoAlias<S>, D: ?Sized> IntoViewMutWithShape<S, D>
    for ViewMut<'_, S2, D>
{
    fn view_mut_with_shape(&mut self, shape: S) -> ViewMut<'_, S, D> {
        self.shape.can_broadcast_into(shape);
        ViewMut {
            shape,
            offset: self.offset,
            strides: S2::into_index(self.strides),
            data: self.data,
        }
    }
}

impl<S: Shape + BroadcastIntoNoAlias<S>, D: ?Sized> IntoViewMut<D> for ViewMut<'_, S, D> {
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
        S: Shape + BroadcastIntoNoAlias<S>,
        S2: Shape + BroadcastIntoNoAlias<S>,
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
        S: Shape + BroadcastIntoNoAlias<S>,
        D: 'a + ?Sized,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > IntoTarget<D> for &'a mut Array<S, D2>
{
    type NativeShape = S;

    fn build(self) -> Self::Target {
        self.view_mut()
    }
}

impl<'a, S: Shape + BroadcastIntoNoAlias<S>, S2: Shape + BroadcastIntoNoAlias<S>, D: ?Sized>
    IntoTargetWithShape<S, D> for &'a mut ViewMut<'a, S2, D>
{
    type Target = ViewMut<'a, S, D>;

    fn build_shape(self, shape: S) -> Self::Target {
        self.view_mut_with_shape(shape)
    }
}

impl<'a, S: Shape + BroadcastIntoNoAlias<S>, D: ?Sized> IntoTarget<D>
    for &'a mut ViewMut<'a, S, D>
{
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
pub struct ArrayTarget<S: Shape, S2: Shape + BroadcastInto<S>, D> {
    array: Array<S2, D>,
    shape: S,
}

impl<
        S: Shape,
        S2: Shape + BroadcastInto<S>,
        S3: Shape + BroadcastInto<S2>,
        D: ?Sized,
        D2: ops::Deref<Target = D>,
    > IntoViewWithShape<S, D> for ArrayTarget<S2, S3, D2>
{
    fn view_with_shape(&self, shape: S) -> View<S, D> {
        self.shape.broadcast_into_fail(shape);
        View {
            shape,
            offset: self.array.offset,
            strides: S2::into_index(S3::into_index(self.array.strides)),
            data: &self.array.data,
        }
    }
}

impl<
        S: Shape + BroadcastInto<S>,
        S2: Shape + BroadcastInto<S>,
        D: ?Sized,
        D2: ops::Deref<Target = D>,
    > IntoView<D> for ArrayTarget<S, S2, D2>
{
    type NativeShape = S;

    fn view(&self) -> View<S, D> {
        View {
            shape: self.shape.clone(),
            offset: self.array.offset,
            strides: S2::into_index(self.array.strides),
            data: &self.array.data,
        }
    }
}

impl<
        S: Shape,
        S2: Shape + BroadcastIntoNoAlias<S>,
        S3: Shape + BroadcastIntoNoAlias<S2>,
        D: ?Sized,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > IntoViewMutWithShape<S, D> for ArrayTarget<S2, S3, D2>
{
    fn view_mut_with_shape(&mut self, shape: S) -> ViewMut<'_, S, D> {
        self.shape.broadcast_into_fail(shape);
        ViewMut {
            shape,
            offset: self.array.offset,
            strides: S2::into_index(S3::into_index(self.array.strides)),
            data: &mut self.array.data,
        }
    }
}

impl<
        S: Shape + BroadcastIntoNoAlias<S>,
        S2: Shape + BroadcastIntoNoAlias<S>,
        D: ?Sized,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > IntoViewMut<D> for ArrayTarget<S, S2, D2>
{
    fn view_mut(&mut self) -> ViewMut<'_, S, D> {
        ViewMut {
            shape: self.shape.clone(),
            offset: self.array.offset,
            strides: S2::into_index(self.array.strides),
            data: &mut self.array.data,
        }
    }
}

impl<'a, S: Shape, S2: Shape + BroadcastIntoNoAlias<S>, D> OutTarget for ArrayTarget<S, S2, D> {
    type Output = Array<S2, D>;

    fn output(self) -> Self::Output {
        self.array
    }
}

impl<
        'a,
        S: Shape + BroadcastIntoNoAlias<S>,
        S2: Shape + BroadcastIntoNoAlias<S>,
        E: Default + Clone,
    > IntoTargetWithShape<S, [E]> for AllocShape<S2, E>
{
    type Target = ArrayTarget<S, S2, Vec<E>>;

    fn build_shape(self, shape: S) -> Self::Target {
        let AllocShape {
            shape: self_shape, ..
        } = self;
        self_shape.broadcast_into_fail(shape);
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

impl<'a, S: Shape + BroadcastIntoNoAlias<S>, E: Default + Clone> IntoTarget<[E]>
    for AllocShape<S, E>
{
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

impl<'a, S: Shape + BroadcastIntoNoAlias<S>, E: Default + Clone, D: ?Sized>
    IntoTargetWithShape<S, D> for Alloc<E>
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
        alloc, alloc_shape, Array, AsIndex, BroadcastTogether, Const, DefiniteRange, IntoTarget,
        IntoTargetWithShape, IntoView, IntoViewMut, NewAxis, OutTarget, Shape, ShapeEq,
    };

    #[test]
    fn test_shapes() {
        let s = (((), Const::<3>), Const::<4>);
        assert!(s.shape_eq(&(((), Const::<3>), Const::<4>)));
        assert!(s.shape_eq(&(((), Const::<3>), 4)));
        assert!(s.shape_eq(&(((), 3), 4)));
        assert!(!s.shape_eq(&(((), Const::<3>), 5)));
        assert!(!s.shape_eq(&(((), 3), 5)));

        assert!((((), Const::<3>), Const::<4>).shape_eq(&s));
        assert!((((), Const::<3>), 4).shape_eq(&s));
        assert!((((), 3), 4).shape_eq(&s));
        assert!(!(((), Const::<3>), 5).shape_eq(&s));
        assert!(!(((), 3), 5).shape_eq(&s));
    }

    #[test]
    fn test_iter() {
        let mut t = Array {
            shape: (((), Const::<2>), Const::<2>),
            strides: (((), Const::<2>), Const::<2>).default_strides(),
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
            shape: (((), Const::<2>), Const::<2>),
            strides: (((), Const::<2>), Const::<2>).default_strides(),
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
        dbg!(bernstein_coef(&a, alloc_shape((((), 2), 2))));
        //panic!();

        //bernstein_coef(&a, &mut b);
    }

    #[test]
    fn test_sum() {
        fn sum(in1: &impl IntoView<[f32]>) -> f32 {
            let in1 = in1.view();

            in1.into_iter().sum()
        }

        let a = Array {
            shape: (((), Const::<2>), Const::<2>),
            strides: (((), Const::<2>), Const::<2>).default_strides(),
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
            shape: (((), Const::<2>), Const::<2>),
            strides: (((), Const::<2>), Const::<2>).default_strides(),
            offset: 0,
            data: vec![1., 2., 3., 4.],
        };

        ones(&mut a);
        ones(alloc_shape((((), 4), 4)));
    }

    #[test]
    fn test_broadcast() {
        let s = (((), Const::<3>), Const::<4>);
        assert_eq!(
            s.broadcast_together((((), Const::<3>), Const::<4>))
                .unwrap()
                .as_index(),
            [3, 4]
        );
        assert_eq!(
            s.broadcast_together((((), Const::<3>), 4))
                .unwrap()
                .as_index(),
            [3, 4]
        );
        assert_eq!(
            s.broadcast_together((((), 3), 4)).unwrap().as_index(),
            [3, 4]
        );
        assert!(s.broadcast_together((((), Const::<3>), 5)).is_none());
        assert!(s.broadcast_together((((), 3), 5)).is_none());

        assert_eq!(s.broadcast_together(()).unwrap().as_index(), [3, 4]);
        assert_eq!(s.broadcast_together(((), 4)).unwrap().as_index(), [3, 4]);
        assert_eq!(
            s.broadcast_together((((), 3), NewAxis)).unwrap().as_index(),
            [3, 4]
        );
        assert_eq!(
            s.broadcast_together(((((), 10), 3), NewAxis))
                .unwrap()
                .as_index(),
            [10, 3, 4]
        );
    }
}
