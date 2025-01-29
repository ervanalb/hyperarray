use std::convert;
use std::error;
use std::fmt;
use std::marker;
use std::ops;

/// A marker used in place of a concrete [Shape]
/// to indicate that any valid shape can be substituted.
/// This is used e.g. for a lazily allocate output array
/// appearing in a collection of parameters that are broadcast together.
#[derive(Clone, Copy, Debug)]
pub struct AnyShape;

/// In a [Shape] tuple, this represents an axis of unit length that can be broadcast to any size.
#[derive(Default, Clone, Copy)]
pub struct NewAxis;

impl fmt::Debug for NewAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_")
    }
}

/// In a [Shape] tuple, this represents an axis length known at compile time.
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

/// An error value that is returned when two shapes could not be broadcast together.
/// This type is generally returned in the `Err` variant of a `Result`
/// if the broadcasting operation has a chance of failing at runtime.
/// If the broadcasting operation cannot fail at runtime, the `Err` type will be [std::convert::Infallible]
#[derive(thiserror::Error, Debug)]
#[error("Broadcast error: axis length mismatch {0} != {1}")]
pub struct BroadcastError(pub usize, pub usize);

impl From<convert::Infallible> for BroadcastError {
    fn from(value: convert::Infallible) -> Self {
        match value {}
    }
}

/// A [Shape] (or placeholder such as [AnyShape])
/// which can be broadcast into the given shape
/// (or placeholder.)
/// For a version of this trait which requires concrete shapes,
/// use [BroadcastIntoConcrete].
pub trait BroadcastInto<T> {
    /// The error type for if the given shape is not compatible at runtime.
    /// If the broadcasting operation cannot fail at runtime, this will be [std::convert::Infallible]
    /// If the broadcasting operation has a chance of failing at runtime,
    /// this will be [BroadcastError].
    type Error: error::Error;

    /// Check if the broadcast can be performed.
    /// If `Ok(())` is returned, it means that `self` can be broadcast to `other`.
    fn check_broadcast_into(self, other: T) -> Result<(), Self::Error>;
}

/// A [Shape] which can be broadcast into the given shape.
/// This specializes [BroadcastInto] for concrete shapes,
/// providing an index conversion method.
pub trait BroadcastIntoConcrete<T: Shape>: Shape + BroadcastInto<T> {
    // Convert an index into an array whose shape is `Self`
    // into an index for an array whose shape is `T`.
    //
    // This function does not depend on the actual shape values,
    // only their types.
    fn convert_index(index: Self::Index) -> T::Index;
}

/// A [Shape] (or placeholder such as [AnyShape])
/// which can be broadcast into the given shape
/// (or placeholder)
/// without aliasing.
/// This generally means that the shapes are equivalent
/// since any non-trivial broadcast creates aliasing.
/// However, this should be preferred for a few reasons:
/// * No two given shapes should be assumed the same type,
///   even if they are equivalent.
///   For instance, `(2, 2)` and `(NewAxis, Const::<2>, Const::<2>)`
///   are not the same type, but are equivalent,
///   and can be broadcast without aliasing.
/// * `AnyShape`, a marker used to indicate that any valid shape can be substituted,
///   can broadcast into any shape without aliasing.
///   This is used e.g. when lazily allocating output arrays.
pub trait BroadcastIntoNoAlias<T>: BroadcastInto<T> {}

/// A [Shape] (or placeholder such as [AnyShape])
/// which can be broadcast with the given shape
/// to form a third, possibly new shape,
/// that is compatible with both.
/// Unlike [BroadcastInto],
/// * this relation is symmetric:
///   `a.broadcast_with(b)` should return the same as `b.broadcast_with(a)`.
/// * this relation may return a shape that is distinct from both of its inputs
///
/// Prefer using [BroadcastTogether].
pub trait BroadcastWith<T: BroadcastInto<Self::Output>>: BroadcastInto<Self::Output> {
    // The type of the resulting shape (or placeholder such as [AnyShape])
    type Output;

    /// The error type for if the given shapes are not compatible at runtime.
    /// If the broadcasting operation cannot fail at runtime, this will be [std::convert::Infallible]
    /// If the broadcasting operation has a chance of failing at runtime,
    /// this will be [BroadcastError].
    type WithError: error::Error;

    // Perform the broadcast
    fn broadcast_with(self, other: T) -> Result<Self::Output, Self::WithError>;
}

/// A set of [Shapes](Shape) (or placeholders such as [AnyShape])
/// which can be broadcast together
/// to form a new shape that is compatible with all.
///
/// Unlike [BroadcastInto],
/// * this relation is symmetric:
///   `(a, b).broadcast_together()` should return the same as `(b, a).broadcast_with()`.
/// * this relation may return a shape that is distinct from all of its inputs
///
/// This is a generalization of [BroadcastWith]
/// to any number of inputs, and should be preferred.
pub trait BroadcastTogether {
    type Output;
    type Error: error::Error;

    fn broadcast_together(self) -> Result<Self::Output, Self::Error>;
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

/// A tuple where each element represents the length of an axis of multi-dimensional data.
/// Each entry may be:
/// * [Const<N>](Const), indicating that the length is known at compile-time
/// * `usize`, indicating that the length is not known at compile-time
/// * [NewAxis], indicating an axis of unit length that can be broadcast to any size.
///
/// // TODO write a shape example
///
/// `NewAxis` is used to build shapes for data that only varies in some axes.
/// For example, a 1D row vector with length 5 may have shape `(5)`,
/// while a 1D column vector with length 5 may have shape `(5, NewAxis)`.
/// In both cases, since the data is 1-dimensional, the [index](Index) would be `[usize; 1]`.
/// But when broadcast together, the result is shape `(5, 5)` with a 2D index.
///
/// Note that unlike NumPy, `1_usize` / `Const<1>` will not broadcast.
///
/// Shapes are assumed to be prefixed
/// with an indefinite set of `NewAxis`.
/// For example, the following shapes are equivalent for all intents and purposes:
/// * `(5)`
/// * `(NewAxis, 5)`
/// * `(NewAxis, NewAxis, 5)`
/// It is discouraged to include `NewAxis` prefixes since they are implied.
///
/// Since there are multiple equivalent `Shape` representations,
/// **no two shapes should be assumed to be the same type**.
/// For instance, `(2, 2)` and `(NewAxis, Const::<2>, Const::<2>)`
/// are not the same type, but are equivalent for all computational purposes.
/// For writing type bounds that enforce shape compatibility, see [Broadcast].
pub trait Shape: 'static + Sized + Clone + Copy + AsIndex + fmt::Debug + CalculateShape {
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
}

/// Represents the coordinates of an element in multi-dimensional data,
/// e.g. a 2D index might be represented as `[usize; 2]`.
/// Each element corresponds to the index along that axis.
///
/// Unlike [Shape], the coordinates of an `Index` are always type `usize`.
/// Every `Shape` type has a corresponding `Index` type.
///
/// Note that unlike NumPy, an array's `Shape` tuple may have more elements than its `Index` array.
/// This is because axes whose length is [NewAxis] are not included in the index.
/// For example, a column vector with shape `(5, NewAxis)` is indexed by `[usize; 1]`
/// (since it is one-dimensional.)
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
    /// The index of the first element in all axes (an index containing all zeros.)
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
    fn iter_mut<'a>(&'a mut self) -> std::slice::IterMut<'a, usize>;
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

/// Build this value into a [View] or similar representation
/// suitable for processing.
///
/// This trait is generally used to help bound generic function parameters.
/// For example, here is a function which takes multi-dimensional data
/// that is in-memory, has element type `f32`, and could be any shape.
///
/// The user may pass:
/// * `&Array`
/// * `&ViewMut`
/// * `&View`
/// at their convenience.
/// ```
/// use nada::{Build, View, Shape};
/// use std::convert::Infallible;
///
/// fn foo<A, S: Shape>(a: &A)
/// where
///     for<'a> &'a A: Build<Output = View<'a, S, [f32]>, Error=Infallible>,
/// {
///     let Ok(a) = a.build();
///
///     // Do stuff with `a`
/// }
/// ```
///
/// In the previous simple example, building one of these types into its native shape is infallible,
/// but more generally, the `build()` call may fail at runtime.
///
/// This second example takes two parameters whose shapes must be broadcastable together.
/// The two [View] instances returned by `build()` will have the same shape `S`,
/// which is the result of the broadcast,
/// allowing them to be zipped or otherwise iterated together.
///
/// ```
/// use nada::{Build, View, Broadcast, Shape};
/// use std::convert::Infallible;
///
/// fn foo<A, B, S: Shape, E>(a: &A, b: &B) -> Result<(), E>
/// where
///     for<'a> Broadcast<(&'a A, &'a B)>: Build<Output = (View<'a, S, [f32]>, View<'a, S, [f32]>), Error=E>,
/// {
///     let (a, b) = Broadcast((a, b)).build()?;
///
///     // Do stuff with `a` and `b`
///     // which are guaranteed to have the same shape
///
///     Ok(())
/// }
/// ```
///
pub trait Build {
    type Output;
    type Error: error::Error;

    // Calculate shapes & construct the resulting view(s) (or other representations.)
    fn build(self) -> Result<Self::Output, Self::Error>;
}

/// Calculate the shape of this value.
///
/// This trait is for internal use only,
/// as part of the two-step [Build] process:
/// 1. Compute shapes (e.g. via broadcasting) (this trait)
/// 2. Create views or other representations with the computed shape ([BuildWithShape])
pub trait CalculateShape {
    type Shape;
    type Error: error::Error;

    fn calculate_shape(&self) -> Result<Self::Shape, Self::Error>;
}

/// Build this value into a [View] or similar representation
/// with the given shape.
///
/// This trait is for internal use only,
/// as part of the two-step [Build] process:
/// 1. Compute shapes (e.g. via broadcasting) ([CalculateShape])
/// 2. Create views or other representations with the computed shape (this trait)
pub trait BuildWithShape<S: Shape> {
    type Output;
    type Error: error::Error;

    fn build_with_shape(self, shape: S) -> Result<Self::Output, Self::Error>;
}

/// A wrapper indicating that the wrapped value should be [built](Build) into an output target.
struct Out<T>(T);

pub struct Broadcast<T>(pub T);

/////////////////////////////////////////////
// Helper trait for fixed-length array building

/// This fixed-length array can be grown by 1 element
pub trait Push {
    type OneBigger: Pop<OneSmaller = Self>;

    /// Return this array with the given element added to the end
    fn append(self, a: usize) -> Self::OneBigger;
}

/// This fixed-length array can be shrunk by 1 element
pub trait Pop {
    type OneSmaller: Push<OneBigger = Self>;

    /// Return the array with its last element split off
    fn split(self) -> (Self::OneSmaller, usize);
}

macro_rules! impl_push_pop {
    ($n:expr, $($i:expr)*) => {
        impl Push for [usize; $n] {
            type OneBigger = [usize; $n+1];

            fn append(self, a: usize) -> Self::OneBigger {
                [$(self[$i],)* a,]
            }
        }

        impl Pop for [usize; $n+1] {
            type OneSmaller = [usize; $n];

            fn split(self) -> ([usize; $n], usize) {
                ([$(self[$i],)*], self[$n])
            }
        }
    };
}

impl_push_pop!(0,);
impl_push_pop!(1, 0);
impl_push_pop!(2, 0 1);
impl_push_pop!(3, 0 1 2);
impl_push_pop!(4, 0 1 2 3);

/// This tuple can be grown by 1 element
pub trait TuplePush {
    type OneBigger<A>: TuplePop<Tail = A, OneSmaller = Self>;

    /// Return this tuple with the given element added to the end
    fn append<A>(self, a: A) -> Self::OneBigger<A>;
}

/// This tuple can be shrunk by 1 element
pub trait TuplePop {
    type Tail;
    type OneSmaller: TuplePush<OneBigger<Self::Tail> = Self>;

    /// Return the tuple with its last element split off
    fn split(self) -> (Self::OneSmaller, Self::Tail);
}

macro_rules! impl_tuple_push_pop {
    ($($a:ident $A:ident)*) => {
        impl<$($A,)*> TuplePush for ($($A,)*) {
            type OneBigger<A> = ($($A,)* A,);

            fn append<A>(self, a: A) -> Self::OneBigger<A> {
                let ($($a,)*) = self;
                ($($a,)* a,)
            }
        }

        impl<$($A,)* A,> TuplePop for ($($A,)* A,) {
            type Tail = A;
            type OneSmaller = ($($A,)*);

            fn split(self) -> (Self::OneSmaller, Self::Tail) {
                let ($($a,)* a,) = self;
                (($($a,)*), a)
            }
        }
    }
}

impl_tuple_push_pop!();
impl_tuple_push_pop!(a0 A0);
impl_tuple_push_pop!(a0 A0 a1 A1);

/////////////////////////////////////////////
// Shape

impl Shape for () {}

macro_rules! impl_shape {
    ($($a:ident $A:ident)*) => {
        impl<$($A,)* const N: usize,> Shape for ($($A,)* Const<N>,)
            where ($($A,)* Const<N>,): 'static + Copy + fmt::Debug + AsIndex + CalculateShape
        {}

        impl<$($A,)*> Shape for ($($A,)* usize,)
            where ($($A,)* usize,): 'static + Copy + fmt::Debug + AsIndex + CalculateShape
        {}

        impl<$($A,)*> Shape for ($($A,)* NewAxis,)
            where ($($A,)* NewAxis,): 'static + Copy + fmt::Debug + AsIndex + CalculateShape
        {}
    };
}

impl_shape!();
impl_shape!(a0 A0);
impl_shape!(a0 A0 a1 A1);

// AsIndex

impl AsIndex for () {
    type Index = [usize; 0];

    fn as_index(&self) -> Self::Index {
        []
    }
}

macro_rules! impl_as_index {
    ($($a:ident $A:ident)*) => {

        impl<$($A: Copy,)* const N: usize, > AsIndex for ($($A,)* Const<N>,)
        where
            ($($A,)*): AsIndex<Index: Push<OneBigger: Index>>,
        {
            type Index = <<($($A,)*) as AsIndex>::Index as Push>::OneBigger;

            fn as_index(&self) -> Self::Index {
                let &($($a,)* _,) = self;
                ($($a,)*).as_index().append(N)
            }
        }


        impl<$($A: Copy,)*> AsIndex for ($($A,)* usize,)
        where
            ($($A,)*): AsIndex<Index: Push<OneBigger: Index>>,
        {
            type Index = <<($($A,)*) as AsIndex>::Index as Push>::OneBigger;

            fn as_index(&self) -> Self::Index {
                let &($($a,)* a,) = self;
                ($($a,)*).as_index().append(a)
            }
        }

        impl<$($A: Copy,)*> AsIndex for ($($A,)* NewAxis,)
        where
            ($($A,)*): AsIndex,
        {
            type Index = <($($A,)*) as AsIndex>::Index;

            fn as_index(&self) -> Self::Index {
                let &($($a,)* _,) = self;
                ($($a,)*).as_index()
            }
        }
    };
}

impl_as_index!();
impl_as_index!(a0 A0);
impl_as_index!(a0 A0 a1 A1);

// CalculateShape

impl CalculateShape for () {
    type Shape = Self;
    type Error = convert::Infallible;

    fn calculate_shape(&self) -> Result<Self::Shape, Self::Error> {
        Ok(*self)
    }
}

macro_rules! impl_shaped {
    ($($a:ident $A:ident)*) => {

        impl<$($A: Copy,)* const N: usize, > CalculateShape for ($($A,)* Const<N>,)
        where
            ($($A,)*): CalculateShape,
        {
            type Shape = Self;
            type Error = convert::Infallible;

            fn calculate_shape(&self) -> Result<Self::Shape, Self::Error> {
                Ok(*self)
            }
        }


        impl<$($A: Copy,)*> CalculateShape for ($($A,)* usize,)
        where
            ($($A,)*): CalculateShape,
        {
            type Shape = Self;
            type Error = convert::Infallible;

            fn calculate_shape(&self) -> Result<Self::Shape, Self::Error> {
                Ok(*self)
            }
        }

        impl<$($A: Copy,)*> CalculateShape for ($($A,)* NewAxis,)
        where
            ($($A,)*): CalculateShape,
        {
            type Shape = Self;
            type Error = convert::Infallible;

            fn calculate_shape(&self) -> Result<Self::Shape, Self::Error> {
                Ok(*self)
            }
        }
    };
}

impl_shaped!();
impl_shaped!(a0 A0);
impl_shaped!(a0 A0 a1 A1);

impl BroadcastInto<()> for () {
    type Error = convert::Infallible;

    fn check_broadcast_into(self, _other: ()) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl BroadcastIntoConcrete<()> for () {
    fn convert_index(_index: [usize; 0]) -> [usize; 0] {
        []
    }
}
impl BroadcastIntoNoAlias<()> for () {}

impl BroadcastInto<AnyShape> for AnyShape {
    type Error = convert::Infallible;
    fn check_broadcast_into(self, _other: AnyShape) -> Result<(), Self::Error> {
        Ok(())
    }
}
impl BroadcastIntoNoAlias<AnyShape> for AnyShape {}

impl<S: Shape> BroadcastInto<S> for AnyShape {
    type Error = convert::Infallible;
    fn check_broadcast_into(self, _other: S) -> Result<(), Self::Error> {
        Ok(())
    }
}
impl<S: Shape> BroadcastIntoNoAlias<S> for AnyShape {}

macro_rules! impl_broadcast_into_a {
    ($($a:ident $A:ident)*) => {
        impl<$($A: Copy + fmt::Debug,)*> BroadcastInto<()> for ($($A,)* NewAxis,)
            where ($($A,)*): BroadcastInto<()>,
        {
            type Error = <($($A,)*) as BroadcastInto<()>>::Error;
            fn check_broadcast_into(self, _other: ()) -> Result<(), Self::Error> {
                let ($($a,)* _,) = self;
                ($($a,)*).check_broadcast_into(())
            }
        }

        impl<$($A: Copy + fmt::Debug,)*> BroadcastIntoNoAlias<()> for ($($A,)* NewAxis,)
            where ($($A,)*): BroadcastIntoConcrete<()>,
            ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
        {}

        impl<$($A: Copy + fmt::Debug,)*> BroadcastIntoConcrete<()> for ($($A,)* NewAxis,)
            where ($($A,)*): BroadcastIntoConcrete<()>,
            ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
        {
            fn convert_index(index: <($($A,)*) as AsIndex>::Index) -> <() as AsIndex>::Index {
                <($($A,)*) as BroadcastIntoConcrete<()>>::convert_index(index)
            }
        }
    };
}

impl_broadcast_into_a!();
impl_broadcast_into_a!(a0 A0);
impl_broadcast_into_a!(a0 A0 a1 A1);

macro_rules! impl_broadcast_into_b {
    ($($b:ident $B:ident)*) => {
        impl<$($B,)*> BroadcastInto<($($B,)* NewAxis,)> for ()
            where (): BroadcastInto<($($B,)*)>,
            ($($B,)*): Shape,
            ($($B,)* NewAxis,): Shape<Index=<($($B,)*) as AsIndex>::Index>,
        {
            type Error = <() as BroadcastInto<($($B,)*)>>::Error;
            fn check_broadcast_into(self, other: ($($B,)* NewAxis,)) -> Result<(), Self::Error> {
                let ($($b,)* _,) = other;
                ().check_broadcast_into(($($b,)*))
            }
        }
        impl<$($B,)*> BroadcastIntoNoAlias<($($B,)* NewAxis,)> for ()
            where (): BroadcastIntoNoAlias<($($B,)*)>,
            ($($B,)*): Shape,
            ($($B,)* NewAxis,): Shape<Index=<($($B,)*) as AsIndex>::Index>,
        {}

        impl<$($B,)*> BroadcastIntoConcrete<($($B,)* NewAxis,)> for ()
            where (): BroadcastIntoConcrete<($($B,)*)>,
            ($($B,)*): Shape,
            ($($B,)* NewAxis,): Shape<Index=<($($B,)*) as AsIndex>::Index>,
        {
            fn convert_index(index: [usize; 0]) -> <($($B,)*) as AsIndex>::Index {
                <() as BroadcastIntoConcrete<($($B,)*)>>::convert_index(index)
            }
        }

        impl<const N: usize, $($B,)*> BroadcastInto<($($B,)* Const<N>,)> for ()
            where (): BroadcastInto<($($B,)*)>,
            ($($B,)*): Shape<Index: Push<OneBigger=<($($B,)* Const<N>,) as AsIndex>::Index>>,
            ($($B,)* Const<N>,): Shape,
        {
            type Error = <() as BroadcastInto<($($B,)*)>>::Error;
            fn check_broadcast_into(self, other: ($($B,)* Const<N>,)) -> Result<(), Self::Error> {
                let ($($b,)* _,) = other;
                ().check_broadcast_into(($($b,)*))
            }
        }
        //impl<const N: usize> BroadcastIntoNoAlias<(S2, Const<N>)> for ()
        //{ ! }
        impl<const N: usize, $($B,)*> BroadcastIntoConcrete<($($B,)* Const<N>,)> for ()
            where (): BroadcastIntoConcrete<($($B,)*)>,
            ($($B,)*): Shape<Index: Push<OneBigger=<($($B,)* Const<N>,) as AsIndex>::Index>>,
            ($($B,)* Const<N>,): Shape,
        {
            fn convert_index(index: [usize; 0]) -> <($($B,)* Const<N>,) as AsIndex>::Index {
                <() as BroadcastIntoConcrete<($($B,)*)>>::convert_index(index).append(0)
            }
        }

        impl<$($B,)*> BroadcastInto<($($B,)* usize,)> for ()
            where (): BroadcastInto<($($B,)*)>,
            ($($B,)*): Shape<Index: Push<OneBigger=<($($B,)* usize,) as AsIndex>::Index>>,
            ($($B,)* usize,): Shape,
        {
            type Error = <() as BroadcastInto<($($B,)*)>>::Error;
            fn check_broadcast_into(self, other: ($($B,)* usize,)) -> Result<(), Self::Error> {
                let ($($b,)* _,) = other;
                ().check_broadcast_into(($($b,)*))
            }
        }
        //impl BroadcastIntoNoAlias<(S2, usize)> for ()
        //{ ! }
        impl<$($B,)*> BroadcastIntoConcrete<($($B,)* usize,)> for ()
            where (): BroadcastIntoConcrete<($($B,)*)>,
            ($($B,)*): Shape<Index: Push<OneBigger=<($($B,)* usize,) as AsIndex>::Index>>,
            ($($B,)* usize,): Shape,
        {
            fn convert_index(index: [usize; 0]) -> <($($B,)* usize,) as AsIndex>::Index {
                <() as BroadcastIntoConcrete<($($B,)*)>>::convert_index(index).append(0)
            }
        }
    };
}

impl_broadcast_into_b!();
impl_broadcast_into_b!(b0 B0);
impl_broadcast_into_b!(b0 B0 b1 B1);

macro_rules! impl_broadcast_into_ab {
    ($($a:ident $A:ident)*, $($b:ident $B:ident)*) => {
        impl<$($A,)* $($B,)*> BroadcastInto<($($B,)* NewAxis,)> for ($($A,)* NewAxis,)
            where ($($A,)*): BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* NewAxis,): Shape<Index=<($($B,)*) as AsIndex>::Index>,
        {
            type Error = <($($A,)*) as BroadcastInto<($($B,)*)>>::Error;
            fn check_broadcast_into(self, other: ($($B,)* NewAxis,)) -> Result<(), Self::Error> {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                ($($a,)*).check_broadcast_into(($($b,)*))
            }

        }
        impl<$($A,)* $($B,)*> BroadcastIntoNoAlias<($($B,)* NewAxis,)> for ($($A,)* NewAxis,)
            where ($($A,)*): BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* NewAxis,): Shape<Index=<($($B,)*) as AsIndex>::Index>,
        {}
        impl<$($A,)* $($B,)*> BroadcastIntoConcrete<($($B,)* NewAxis,)> for ($($A,)* NewAxis,)
            where ($($A,)*): BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* NewAxis,): Shape<Index=<($($B,)*) as AsIndex>::Index>,
        {
            fn convert_index(index: <($($A,)* NewAxis,) as AsIndex>::Index) -> <($($B,)*) as AsIndex>::Index {
                <($($A,)*) as BroadcastIntoConcrete<($($B,)*)>>::convert_index(index)
            }
        }

        impl<const N: usize, $($A,)* $($B,)*> BroadcastInto<($($B,)* Const<N>,)> for ($($A,)* NewAxis,)
            where ($($A,)*): BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            type Error = <($($A,)*) as BroadcastInto<($($B,)*)>>::Error;
            fn check_broadcast_into(self, other: ($($B,)* Const<N>,)) -> Result<(), Self::Error> {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                ($($a,)*).check_broadcast_into(($($b,)*))
            }
        }
        //impl<const N: usize, $($A,)* $($B,)*> BroadcastInto<($($B,)* Const<N>,)> for ($($A,)* NewAxis,) { ! }
        impl<const N: usize, $($A,)* $($B,)*> BroadcastIntoConcrete<($($B,)* Const<N>,)> for ($($A,)* NewAxis,)
            where ($($A,)*): BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn convert_index(index: <($($A,)* NewAxis,) as AsIndex>::Index) -> <($($B,)* Const<N>,) as AsIndex>::Index {
                <($($A,)*) as BroadcastIntoConcrete<($($B,)*)>>::convert_index(index).append(0)
            }
        }

        impl<$($A,)* $($B,)*> BroadcastInto<($($B,)* usize,)> for ($($A,)* NewAxis,)
            where ($($A,)*): BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            type Error = <($($A,)*) as BroadcastInto<($($B,)*)>>::Error;
            fn check_broadcast_into(self, other: ($($B,)* usize,)) -> Result<(), Self::Error> {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                ($($a,)*).check_broadcast_into(($($b,)*))
            }
        }
        //impl<$($A,)* $($B,)*> BroadcastInto<($($B,)* Const<N>,)> for ($($A,)* NewAxis,) { ! }
        impl<$($A,)* $($B,)*> BroadcastIntoConcrete<($($B,)* usize,)> for ($($A,)* NewAxis,)
            where ($($A,)*): BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn convert_index(index: <($($A,)* NewAxis,) as AsIndex>::Index) -> <($($B,)* usize,) as AsIndex>::Index {
                <($($A,)*) as BroadcastIntoConcrete<($($B,)*)>>::convert_index(index).append(0)
            }
        }

        impl<const N: usize, $($A,)* $($B,)*> BroadcastInto<($($B,)* Const<N>,)> for ($($A,)* Const<N>,)
            where ($($A,)*): Shape<Index: Push> + BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* Const<N>,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            type Error = <($($A,)*) as BroadcastInto<($($B,)*)>>::Error;
            fn check_broadcast_into(self, other: ($($B,)* Const<N>,)) -> Result<(), Self::Error> {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                ($($a,)*).check_broadcast_into(($($b,)*))
            }
        }
        impl<const N: usize, $($A,)* $($B,)*> BroadcastIntoNoAlias<($($B,)* Const<N>,)> for ($($A,)* Const<N>,)
            where ($($A,)*): Shape<Index: Push> + BroadcastIntoConcrete<($($B,)*)> + BroadcastIntoNoAlias<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* Const<N>,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {}
        impl<const N: usize, $($A,)* $($B,)*> BroadcastIntoConcrete<($($B,)* Const<N>,)> for ($($A,)* Const<N>,)
            where ($($A,)*): Shape<Index: Push> + BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* Const<N>,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn convert_index(index: <($($A,)* Const<N>,) as AsIndex>::Index) -> <($($B,)* Const<N>,) as AsIndex>::Index {
                let (index_rest, index_last) = index.split();
                <($($A,)*) as BroadcastIntoConcrete<($($B,)*)>>::convert_index(index_rest).append(index_last)
            }
        }

        impl<const N: usize, $($A,)* $($B,)*> BroadcastInto<($($B,)* usize,)> for ($($A,)* Const<N>,)
            where ($($A,)*): Shape<Index: Push> + BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* Const<N>,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
                  BroadcastError: From<<($($A,)*) as BroadcastInto<($($B,)*)>>::Error>,
        {
            type Error = BroadcastError;
            fn check_broadcast_into(self, other: ($($B,)* usize,)) -> Result<(), Self::Error> {
                let ($($a,)* _,) = self;
                let ($($b,)* b,) = other;
                ($($a,)*).check_broadcast_into(($($b,)*))?;
                if N != b {
                    return Err(BroadcastError(N, b));
                }
                Ok(())
            }
        }
        impl<const N: usize, $($A,)* $($B,)*> BroadcastIntoNoAlias<($($B,)* usize,)> for ($($A,)* Const<N>,)
            where ($($A,)*): Shape<Index: Push> + BroadcastIntoConcrete<($($B,)*)> + BroadcastIntoNoAlias<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* Const<N>,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
                  BroadcastError: From<<($($A,)*) as BroadcastInto<($($B,)*)>>::Error>,
        {}
        impl<const N: usize, $($A,)* $($B,)*> BroadcastIntoConcrete<($($B,)* usize,)> for ($($A,)* Const<N>,)
            where ($($A,)*): Shape<Index: Push> + BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* Const<N>,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
                  BroadcastError: From<<($($A,)*) as BroadcastInto<($($B,)*)>>::Error>,
        {
            fn convert_index(index: <($($A,)* Const<N>,) as AsIndex>::Index) -> <($($B,)* usize,) as AsIndex>::Index {
                let (index_rest, index_last) = index.split();
                <($($A,)*) as BroadcastIntoConcrete<($($B,)*)>>::convert_index(index_rest).append(index_last)
            }
        }

        impl<const N: usize, $($A,)* $($B,)*> BroadcastInto<($($B,)* Const<N>,)> for ($($A,)* usize,)
            where ($($A,)*): Shape<Index: Push> + BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* usize,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
                  BroadcastError: From<<($($A,)*) as BroadcastInto<($($B,)*)>>::Error>,
        {
            type Error = BroadcastError;
            fn check_broadcast_into(self, other: ($($B,)* Const<N>,)) -> Result<(), Self::Error> {
                let ($($a,)* a,) = self;
                let ($($b,)* _,) = other;
                ($($a,)*).check_broadcast_into(($($b,)*))?;
                if a != N {
                    return Err(BroadcastError(a, N));
                }
                Ok(())
            }
        }
        impl<const N: usize, $($A,)* $($B,)*> BroadcastIntoNoAlias<($($B,)* Const<N>,)> for ($($A,)* usize,)
            where ($($A,)*): Shape<Index: Push> + BroadcastIntoConcrete<($($B,)*)> + BroadcastIntoNoAlias<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* usize,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
                  BroadcastError: From<<($($A,)*) as BroadcastInto<($($B,)*)>>::Error>,
        {}
        impl<const N: usize, $($A,)* $($B,)*> BroadcastIntoConcrete<($($B,)* Const<N>,)> for ($($A,)* usize,)
            where ($($A,)*): Shape<Index: Push> + BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* usize,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
                  BroadcastError: From<<($($A,)*) as BroadcastInto<($($B,)*)>>::Error>,
        {
            fn convert_index(index: <($($A,)* usize,) as AsIndex>::Index) -> <($($B,)* Const<N>,) as AsIndex>::Index {
                let (index_rest, index_last) = index.split();
                <($($A,)*) as BroadcastIntoConcrete<($($B,)*)>>::convert_index(index_rest).append(index_last)
            }
        }

        impl<$($A,)* $($B,)*> BroadcastInto<($($B,)* usize,)> for ($($A,)* usize,)
            where ($($A,)*): Shape<Index: Push> + BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* usize,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
                  BroadcastError: From<<($($A,)*) as BroadcastInto<($($B,)*)>>::Error>,
        {
            type Error = BroadcastError;
            fn check_broadcast_into(self, other: ($($B,)* usize,)) -> Result<(), Self::Error> {
                let ($($a,)* a,) = self;
                let ($($b,)* b,) = other;
                ($($a,)*).check_broadcast_into(($($b,)*))?;
                if a != b {
                    return Err(BroadcastError(a, b));
                }
                Ok(())
            }
        }
        impl<$($A,)* $($B,)*> BroadcastIntoNoAlias<($($B,)* usize,)> for ($($A,)* usize,)
            where ($($A,)*): Shape<Index: Push> + BroadcastIntoConcrete<($($B,)*)> + BroadcastIntoNoAlias<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* usize,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
                  BroadcastError: From<<($($A,)*) as BroadcastInto<($($B,)*)>>::Error>,
        {}
        impl<$($A,)* $($B,)*> BroadcastIntoConcrete<($($B,)* usize,)> for ($($A,)* usize,)
            where ($($A,)*): Shape<Index: Push> + BroadcastIntoConcrete<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* usize,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
                  BroadcastError: From<<($($A,)*) as BroadcastInto<($($B,)*)>>::Error>,
        {
            fn convert_index(index: <($($A,)* usize,) as AsIndex>::Index) -> <($($B,)* usize,) as AsIndex>::Index {
                let (index_rest, index_last) = index.split();
                <($($A,)*) as BroadcastIntoConcrete<($($B,)*)>>::convert_index(index_rest).append(index_last)
            }
        }
    };
}

impl_broadcast_into_ab!(,);
impl_broadcast_into_ab!(a0 A0,);
impl_broadcast_into_ab!(a0 A0 a1 A1,);
impl_broadcast_into_ab!(,b0 B0);
impl_broadcast_into_ab!(a0 A0,b0 B0);
impl_broadcast_into_ab!(a0 A0 a1 A1,b0 B0);
impl_broadcast_into_ab!(,b0 B0 b1 B1);
impl_broadcast_into_ab!(a0 A0,b0 B0 b1 B1);
impl_broadcast_into_ab!(a0 A0 a1 A1,b0 B0 b1 B1);

// BroadcastWith

macro_rules! impl_broadcast_with_a {
    () => {
        impl BroadcastWith<()> for () {
            type Output = ();
            type WithError = convert::Infallible;

            fn broadcast_with(self, _other: ()) -> Result<Self::Output, Self::WithError> {
                Ok(())
            }
        }
        impl BroadcastWith<AnyShape> for AnyShape {
            type Output = AnyShape;
            type WithError = convert::Infallible;

            fn broadcast_with(self, _other: AnyShape) -> Result<Self::Output, Self::WithError> {
                Ok(AnyShape)
            }
        }
        impl BroadcastWith<()> for AnyShape {
            type Output = ();
            type WithError = convert::Infallible;

            fn broadcast_with(self, _other: ()) -> Result<Self::Output, Self::WithError> {
                Ok(())
            }
        }
        impl BroadcastWith<AnyShape> for () {
            type Output = ();
            type WithError = convert::Infallible;

            fn broadcast_with(self, _other: AnyShape) -> Result<Self::Output, Self::WithError> {
                Ok(())
            }
        }
    };

    ($($a:ident $A:ident)*) => {
        impl<$($A,)*> BroadcastWith<()> for ($($A,)*)
            where (): BroadcastInto<($($A,)*)>,
                  ($($A,)*): BroadcastInto<($($A,)*)>,
        {
            type Output = ($($A,)*);
            type WithError = convert::Infallible;

            fn broadcast_with(self, _other: ()) -> Result<Self::Output, Self::WithError> {
                Ok(self)
            }
        }
        impl<$($A,)*> BroadcastWith<AnyShape> for ($($A,)*)
            where AnyShape: BroadcastInto<($($A,)*)>,
                  ($($A,)*): BroadcastInto<($($A,)*)>,
        {
            type Output = ($($A,)*);
            type WithError = convert::Infallible;

            fn broadcast_with(self, _other: AnyShape) -> Result<Self::Output, Self::WithError> {
                Ok(self)
            }
        }


        impl<$($A,)*> BroadcastWith<($($A,)*)> for ()
            where (): BroadcastInto<($($A,)*)>,
                  ($($A,)*): BroadcastInto<($($A,)*)>,
        {
            type Output = ($($A,)*);
            type WithError = convert::Infallible;

            fn broadcast_with(self, other: ($($A,)*)) -> Result<Self::Output, Self::WithError> {
                Ok(other)
            }
        }
        impl<$($A,)*> BroadcastWith<($($A,)*)> for AnyShape
            where AnyShape: BroadcastInto<($($A,)*)>,
                  ($($A,)*): BroadcastInto<($($A,)*)>,
        {
            type Output = ($($A,)*);
            type WithError = convert::Infallible;

            fn broadcast_with(self, other: ($($A,)*)) -> Result<Self::Output, Self::WithError> {
                Ok(other)
            }
        }
    };
}

impl_broadcast_with_a!();
impl_broadcast_with_a!(a0 A0);
impl_broadcast_with_a!(a0 A0 a1 A1);
impl_broadcast_with_a!(a0 A0 a1 A1 a2 A2);

macro_rules! impl_broadcast_with_ab {
    ($($a:ident $A:ident)*, $($b:ident $B:ident)*) => {

        impl<$($A,)* $($B,)* O,> BroadcastWith<($($B,)* NewAxis,)> for ($($A,)* NewAxis,)
            where O: Shape,
                  ($($A,)*): BroadcastWith<($($B,)*), Output: TuplePush<OneBigger<NewAxis>=O>>,
                  ($($B,)*): BroadcastInto<<($($A,)*) as BroadcastWith<($($B,)*)>>::Output>,
                  ($($A,)* NewAxis,): BroadcastInto<O>,
                  ($($B,)* NewAxis,): BroadcastInto<O>,
        {
            type Output = O;
            type WithError = <($($A,)*) as BroadcastWith<($($B,)*)>>::WithError;

            fn broadcast_with(self, other: ($($B,)* NewAxis,)) -> Result<Self::Output, Self::WithError> {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                Ok(($($a,)*).broadcast_with(($($b,)*))?.append(NewAxis))
            }
        }

        impl<const N: usize, $($A,)* $($B,)* O,> BroadcastWith<($($B,)* Const<N>,)> for ($($A,)* NewAxis,)
            where O: Shape,
                  ($($A,)*): BroadcastWith<($($B,)*), Output: TuplePush<OneBigger<Const<N>>=O>>,
                  ($($B,)*): BroadcastInto<<($($A,)*) as BroadcastWith<($($B,)*)>>::Output>,
                  ($($A,)* NewAxis,): BroadcastInto<O>,
                  ($($B,)* Const<N>,): BroadcastInto<O>,
        {
            type Output = O;
            type WithError = <($($A,)*) as BroadcastWith<($($B,)*)>>::WithError;

            fn broadcast_with(self, other: ($($B,)* Const<N>,)) -> Result<Self::Output, Self::WithError> {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                Ok(($($a,)*).broadcast_with(($($b,)*))?.append(Const))
            }
        }

        impl<$($A,)* $($B,)* O,> BroadcastWith<($($B,)* usize,)> for ($($A,)* NewAxis,)
            where O: Shape,
                  ($($A,)*): BroadcastWith<($($B,)*), Output: TuplePush<OneBigger<usize>=O>>,
                  ($($B,)*): BroadcastInto<<($($A,)*) as BroadcastWith<($($B,)*)>>::Output>,
                  ($($A,)* NewAxis,): BroadcastInto<O>,
                  ($($B,)* usize,): BroadcastInto<O>,
        {
            type Output = O;
            type WithError = <($($A,)*) as BroadcastWith<($($B,)*)>>::WithError;

            fn broadcast_with(self, other: ($($B,)* usize,)) -> Result<Self::Output, Self::WithError> {
                let ($($a,)* _,) = self;
                let ($($b,)* b,) = other;
                Ok(($($a,)*).broadcast_with(($($b,)*))?.append(b))
            }
        }

        impl<const N: usize, $($A,)* $($B,)* O,> BroadcastWith<($($B,)* NewAxis,)> for ($($A,)* Const<N>,)
            where O: Shape,
                  ($($A,)*): BroadcastWith<($($B,)*), Output: TuplePush<OneBigger<Const<N>>=O>>,
                  ($($B,)*): BroadcastInto<<($($A,)*) as BroadcastWith<($($B,)*)>>::Output>,
                  ($($A,)* Const<N>,): BroadcastInto<O>,
                  ($($B,)* NewAxis,): BroadcastInto<O>,
        {
            type Output = O;
            type WithError = <($($A,)*) as BroadcastWith<($($B,)*)>>::WithError;

            fn broadcast_with(self, other: ($($B,)* NewAxis,)) -> Result<Self::Output, Self::WithError> {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                Ok(($($a,)*).broadcast_with(($($b,)*))?.append(Const))
            }
        }

        impl<const N: usize, $($A,)* $($B,)* O,> BroadcastWith<($($B,)* Const<N>,)> for ($($A,)* Const<N>,)
            where O: Shape,
                  ($($A,)*): BroadcastWith<($($B,)*), Output: TuplePush<OneBigger<Const<N>>=O>>,
                  ($($B,)*): BroadcastInto<<($($A,)*) as BroadcastWith<($($B,)*)>>::Output>,
                  ($($A,)* Const<N>,): BroadcastInto<O>,
                  ($($B,)* Const<N>,): BroadcastInto<O>,
        {
            type Output = O;
            type WithError = <($($A,)*) as BroadcastWith<($($B,)*)>>::WithError;

            fn broadcast_with(self, other: ($($B,)* Const<N>,)) -> Result<Self::Output, Self::WithError> {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                Ok(($($a,)*).broadcast_with(($($b,)*))?.append(Const))
            }
        }

        impl<const N: usize, $($A,)* $($B,)* O,> BroadcastWith<($($B,)* usize,)> for ($($A,)* Const<N>,)
            where O: Shape,
                  ($($A,)*): BroadcastWith<($($B,)*), Output: TuplePush<OneBigger<Const<N>>=O>>,
                  ($($B,)*): BroadcastInto<<($($A,)*) as BroadcastWith<($($B,)*)>>::Output>,
                  ($($A,)* Const<N>,): BroadcastInto<O>,
                  ($($B,)* usize,): BroadcastInto<O>,
                  BroadcastError: From<<($($A,)*) as BroadcastWith<($($B,)*)>>::WithError>,
        {
            type Output = O;
            type WithError = BroadcastError;

            fn broadcast_with(self, other: ($($B,)* usize,)) -> Result<Self::Output, Self::WithError> {
                let ($($a,)* _,) = self;
                let ($($b,)* b,) = other;
                if N != b {
                    return Err(BroadcastError(N, b));
                }
                Ok(($($a,)*).broadcast_with(($($b,)*))?.append(Const))
            }
        }

        impl<$($A,)* $($B,)* O,> BroadcastWith<($($B,)* NewAxis,)> for ($($A,)* usize,)
            where O: Shape,
                  ($($A,)*): BroadcastWith<($($B,)*), Output: TuplePush<OneBigger<usize>=O>>,
                  ($($B,)*): BroadcastInto<<($($A,)*) as BroadcastWith<($($B,)*)>>::Output>,
                  ($($A,)* usize,): BroadcastInto<O>,
                  ($($B,)* NewAxis,): BroadcastInto<O>,
        {
            type Output = O;
            type WithError = <($($A,)*) as BroadcastWith<($($B,)*)>>::WithError;

            fn broadcast_with(self, other: ($($B,)* NewAxis,)) -> Result<Self::Output, Self::WithError> {
                let ($($a,)* a,) = self;
                let ($($b,)* _,) = other;
                Ok(($($a,)*).broadcast_with(($($b,)*))?.append(a))
            }
        }

        impl<const N: usize, $($A,)* $($B,)* O,> BroadcastWith<($($B,)* Const<N>,)> for ($($A,)* usize,)
            where O: Shape,
                  ($($A,)*): BroadcastWith<($($B,)*), Output: TuplePush<OneBigger<Const<N>>=O>>,
                  ($($B,)*): BroadcastInto<<($($A,)*) as BroadcastWith<($($B,)*)>>::Output>,
                  ($($A,)* usize,): BroadcastInto<O>,
                  ($($B,)* Const<N>,): BroadcastInto<O>,
                  BroadcastError: From<<($($A,)*) as BroadcastWith<($($B,)*)>>::WithError>,
        {
            type Output = O;
            type WithError = BroadcastError;

            fn broadcast_with(self, other: ($($B,)* Const<N>,)) -> Result<Self::Output, Self::WithError> {
                let ($($a,)* a,) = self;
                let ($($b,)* _,) = other;
                if a != N {
                    return Err(BroadcastError(a, N));
                }
                Ok(($($a,)*).broadcast_with(($($b,)*))?.append(Const))
            }
        }

        impl<$($A,)* $($B,)* O,> BroadcastWith<($($B,)* usize,)> for ($($A,)* usize,)
            where O: Shape,
                  ($($A,)*): BroadcastWith<($($B,)*), Output: TuplePush<OneBigger<usize>=O>>,
                  ($($B,)*): BroadcastInto<<($($A,)*) as BroadcastWith<($($B,)*)>>::Output>,
                  ($($A,)* usize,): BroadcastInto<O>,
                  ($($B,)* usize,): BroadcastInto<O>,
                  BroadcastError: From<<($($A,)*) as BroadcastWith<($($B,)*)>>::WithError>,
        {
            type Output = O;
            type WithError = BroadcastError;

            fn broadcast_with(self, other: ($($B,)* usize,)) -> Result<Self::Output, Self::WithError> {
                let ($($a,)* a,) = self;
                let ($($b,)* b,) = other;
                if a != b {
                    return Err(BroadcastError(a, b));
                }
                Ok(($($a,)*).broadcast_with(($($b,)*))?.append(a))
            }
        }
    };
}

impl_broadcast_with_ab!(,);
impl_broadcast_with_ab!(a0 A0,);
impl_broadcast_with_ab!(a0 A0 a1 A1,);
impl_broadcast_with_ab!(,b0 B0);
impl_broadcast_with_ab!(a0 A0,b0 B0);
impl_broadcast_with_ab!(a0 A0 a1 A1,b0 B0);
impl_broadcast_with_ab!(,b0 B0 b1 B1);
impl_broadcast_with_ab!(a0 A0,b0 B0 b1 B1);
impl_broadcast_with_ab!(a0 A0 a1 A1,b0 B0 b1 B1);

// BroadcastTogether

impl BroadcastTogether for () {
    type Output = ();
    type Error = convert::Infallible;

    fn broadcast_together(self) -> Result<Self::Output, Self::Error> {
        Ok(())
    }
}

macro_rules! impl_broadcast_together {
    ($($a:ident $A:ident)*) => {
        impl<$($A,)* A, O> BroadcastTogether for ($($A,)* A,)
        where
            ($($A,)*): BroadcastTogether<Output=O>,
            O: BroadcastWith<A, WithError: From<<($($A,)*) as BroadcastTogether>::Error>>,
            A: BroadcastInto<<O as BroadcastWith<A>>::Output>,
        {
            type Output = <O as BroadcastWith<A>>::Output;
            type Error = <O as BroadcastWith<A>>::WithError;

            fn broadcast_together(self) -> Result<Self::Output, Self::Error> {
                let ($($a,)* a,) = self;
                Ok(($($a,)*).broadcast_together()?.broadcast_with(a)?)
            }
        }
    };
}

impl_broadcast_together!();
impl_broadcast_together!(a0 A0);
impl_broadcast_together!(a0 A0 a1 A1);
impl_broadcast_together!(a0 A0 a1 A1 a2 A2);

impl<S: Shape, D: ?Sized, D2: ops::Deref<Target = D>> CalculateShape for &Array<S, D2> {
    type Shape = S;
    type Error = convert::Infallible;

    fn calculate_shape(&self) -> Result<Self::Shape, Self::Error> {
        Ok(self.shape)
    }
}

impl<S: Shape, D: ?Sized> CalculateShape for &View<'_, S, D> {
    type Shape = S;
    type Error = convert::Infallible;

    fn calculate_shape(&self) -> Result<Self::Shape, Self::Error> {
        Ok(self.shape)
    }
}

impl<S: Shape, D: ?Sized> CalculateShape for &ViewMut<'_, S, D> {
    type Shape = S;
    type Error = convert::Infallible;

    fn calculate_shape(&self) -> Result<Self::Shape, Self::Error> {
        Ok(self.shape)
    }
}

impl<'a, S: Shape, D: ?Sized + 'a, D2: ops::Deref<Target = D>> Build for &'a Array<S, D2> {
    type Output = View<'a, S, D>;
    type Error = convert::Infallible;

    fn build(self) -> Result<Self::Output, Self::Error> {
        Ok(View {
            shape: self.shape,
            offset: self.offset,
            strides: self.strides,
            data: &self.data,
        })
    }
}

impl<'a, S: Shape, S2: BroadcastIntoConcrete<S>, D: ?Sized + 'a, D2: ops::Deref<Target = D>>
    BuildWithShape<S> for &'a Array<S2, D2>
{
    type Output = View<'a, S, D>;
    type Error = <S2 as BroadcastInto<S>>::Error;

    fn build_with_shape(self, shape: S) -> Result<Self::Output, Self::Error> {
        self.shape.check_broadcast_into(shape)?;
        Ok(View {
            shape,
            offset: self.offset,
            strides: S2::convert_index(self.strides),
            data: &self.data,
        })
    }
}

impl<'a, S: Shape, D: ?Sized> Build for &'a View<'a, S, D> {
    type Output = View<'a, S, D>;
    type Error = convert::Infallible;

    fn build(self) -> Result<Self::Output, Self::Error> {
        Ok(View {
            shape: self.shape,
            offset: self.offset,
            strides: self.strides,
            data: self.data,
        })
    }
}

impl<'a, S: Shape, S2: BroadcastIntoConcrete<S>, D: ?Sized> BuildWithShape<S>
    for &'a View<'a, S2, D>
{
    type Output = View<'a, S, D>;
    type Error = <S2 as BroadcastInto<S>>::Error;

    fn build_with_shape(self, shape: S) -> Result<Self::Output, Self::Error> {
        self.shape.check_broadcast_into(shape)?;
        Ok(View {
            shape,
            offset: self.offset,
            strides: S2::convert_index(self.strides),
            data: self.data,
        })
    }
}

impl<'a, S: Shape, D: ?Sized> Build for &'a ViewMut<'a, S, D> {
    type Output = View<'a, S, D>;
    type Error = convert::Infallible;

    fn build(self) -> Result<Self::Output, Self::Error> {
        Ok(View {
            shape: self.shape,
            offset: self.offset,
            strides: self.strides,
            data: self.data,
        })
    }
}

impl<'a, S: Shape, S2: BroadcastIntoConcrete<S>, D: ?Sized> BuildWithShape<S>
    for &'a ViewMut<'a, S2, D>
{
    type Output = View<'a, S, D>;
    type Error = <S2 as BroadcastInto<S>>::Error;

    fn build_with_shape(self, shape: S) -> Result<Self::Output, Self::Error> {
        self.shape.check_broadcast_into(shape)?;
        Ok(View {
            shape,
            offset: self.offset,
            strides: S2::convert_index(self.strides),
            data: self.data,
        })
    }
}

impl<'a, S: Shape, D: ?Sized + 'a, D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>> Build
    for &'a mut Array<S, D2>
{
    type Output = ViewMut<'a, S, D>;
    type Error = convert::Infallible;

    fn build(self) -> Result<Self::Output, Self::Error> {
        Ok(ViewMut {
            shape: self.shape.clone(),
            offset: self.offset,
            strides: self.strides.clone(),
            data: &mut self.data,
        })
    }
}

impl<
        'a,
        S: Shape,
        S2: BroadcastIntoConcrete<S> + BroadcastIntoNoAlias<S>,
        D: ?Sized + 'a,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > BuildWithShape<S> for &'a mut Array<S2, D2>
{
    type Output = ViewMut<'a, S, D>;
    type Error = <S2 as BroadcastInto<S>>::Error;

    fn build_with_shape(self, shape: S) -> Result<Self::Output, Self::Error> {
        self.shape.check_broadcast_into(shape)?;
        Ok(ViewMut {
            shape,
            offset: self.offset,
            strides: S2::convert_index(self.strides),
            data: &mut self.data,
        })
    }
}

impl<'a, S: Shape, D: ?Sized> Build for &'a mut ViewMut<'a, S, D> {
    type Output = ViewMut<'a, S, D>;
    type Error = convert::Infallible;

    fn build(self) -> Result<Self::Output, Self::Error> {
        Ok(ViewMut {
            shape: self.shape,
            offset: self.offset,
            strides: self.strides,
            data: self.data,
        })
    }
}

impl<'a, S: Shape, S2: BroadcastIntoConcrete<S> + BroadcastIntoNoAlias<S>, D: ?Sized>
    BuildWithShape<S> for &'a mut ViewMut<'a, S2, D>
{
    type Output = ViewMut<'a, S, D>;
    type Error = <S2 as BroadcastInto<S>>::Error;

    fn build_with_shape(self, shape: S) -> Result<Self::Output, Self::Error> {
        self.shape.check_broadcast_into(shape)?;
        Ok(ViewMut {
            shape,
            offset: self.offset,
            strides: S2::convert_index(self.strides),
            data: self.data,
        })
    }
}

//////////////////////////////////////////////////////

impl<'a, S: Shape, D: 'a + ?Sized, D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>>
    CalculateShape for Out<&'a mut Array<S, D2>>
{
    type Shape = S;
    type Error = convert::Infallible;

    fn calculate_shape(&self) -> Result<Self::Shape, Self::Error> {
        Ok(self.0.shape)
    }
}

impl<'a, S: Shape, D: 'a + ?Sized, D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>> Build
    for Out<&'a mut Array<S, D2>>
{
    type Output = ViewMut<'a, S, D>;
    type Error = <&'a mut Array<S, D2> as Build>::Error;

    fn build(self) -> Result<Self::Output, Self::Error> {
        self.0.build()
    }
}

impl<
        'a,
        S: Shape,
        S2: BroadcastIntoConcrete<S> + BroadcastIntoNoAlias<S>,
        D: 'a + ?Sized,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > BuildWithShape<S> for Out<&'a mut Array<S2, D2>>
{
    type Output = ViewMut<'a, S, D>;
    type Error = <&'a mut Array<S2, D2> as BuildWithShape<S>>::Error;

    fn build_with_shape(self, shape: S) -> Result<Self::Output, Self::Error> {
        self.0.build_with_shape(shape)
    }
}

impl<'a, S: Shape, D: 'a + ?Sized> CalculateShape for Out<&'a mut ViewMut<'a, S, D>> {
    type Shape = S;
    type Error = convert::Infallible;

    fn calculate_shape(&self) -> Result<Self::Shape, Self::Error> {
        Ok(self.0.shape)
    }
}

impl<'a, S: Shape, D: 'a + ?Sized> Build for Out<&'a mut ViewMut<'a, S, D>> {
    type Output = ViewMut<'a, S, D>;
    type Error = <&'a mut ViewMut<'a, S, D> as Build>::Error;

    fn build(self) -> Result<Self::Output, Self::Error> {
        self.0.build()
    }
}

impl<'a, S: Shape, S2: BroadcastIntoConcrete<S> + BroadcastIntoNoAlias<S>, D: 'a + ?Sized>
    BuildWithShape<S> for Out<&'a mut ViewMut<'a, S2, D>>
{
    type Output = ViewMut<'a, S, D>;
    type Error = <&'a mut ViewMut<'a, S2, D> as BuildWithShape<S>>::Error;

    fn build_with_shape(self, shape: S) -> Result<Self::Output, Self::Error> {
        self.0.build_with_shape(shape)
    }
}

impl<'a, S: Shape, E: Default + Clone> CalculateShape for Out<AllocShape<S, E>> {
    type Shape = S;
    type Error = convert::Infallible;

    fn calculate_shape(&self) -> Result<Self::Shape, Self::Error> {
        Ok(self.0.shape)
    }
}

impl<'a, E: Default + Clone> CalculateShape for Out<Alloc<E>> {
    type Shape = AnyShape;
    type Error = convert::Infallible;

    fn calculate_shape(&self) -> Result<Self::Shape, Self::Error> {
        Ok(AnyShape)
    }
}

impl<'a, S: Shape, E: Default + Clone> Build for Out<AllocShape<S, E>> {
    type Output = Array<S, Vec<E>>;
    type Error = convert::Infallible;

    fn build(self) -> Result<Self::Output, Self::Error> {
        let Out(AllocShape { shape, .. }) = self;
        Ok(Array {
            shape,
            strides: shape.default_strides(),
            offset: 0,
            data: vec![E::default(); shape.num_elements()],
        })
    }
}

// Can't Build for Alloc since no shape information is available

impl<'a, S: Shape, S2: Shape + BroadcastIntoNoAlias<S>, E: Default + Clone> BuildWithShape<S>
    for Out<AllocShape<S2, E>>
{
    type Output = ArrayTarget<S, S2, Vec<E>>;
    type Error = <S2 as BroadcastInto<S>>::Error;

    fn build_with_shape(self, shape: S) -> Result<Self::Output, Self::Error> {
        let Out(AllocShape {
            shape: self_shape, ..
        }) = self;
        self_shape.check_broadcast_into(shape)?;
        Ok(ArrayTarget {
            shape,
            array: Array {
                shape: self_shape,
                strides: self_shape.default_strides(),
                offset: 0,
                data: vec![E::default(); self_shape.num_elements()],
            },
        })
    }
}

impl<'a, S: Shape, E: Default + Clone> BuildWithShape<S> for Out<Alloc<E>> {
    type Output = Array<S, Vec<E>>;
    type Error = convert::Infallible;

    fn build_with_shape(self, shape: S) -> Result<Self::Output, Self::Error> {
        Ok(Array {
            shape,
            strides: shape.default_strides(),
            offset: 0,
            data: vec![E::default(); shape.num_elements()],
        })
    }
}

macro_rules! impl_broadcast_builder {
    () => {
        impl Build for Broadcast<()>
        {
            type Output = ();
            type Error = convert::Infallible;

            fn build(self) -> Result<Self::Output, Self::Error> {
                Ok(())
            }
        }

        impl<S: Shape> BuildWithShape<S> for Broadcast<()>
            where (): BroadcastInto<S>
        {
            type Output = ();
            type Error = <() as BroadcastInto<S>>::Error;

            fn build_with_shape(self, shape: S) -> Result<Self::Output, Self::Error> {
                ().check_broadcast_into(shape)
            }
        }
    };

    ($($a:ident $A:ident)*) => {
        impl<$($A,)* O: Shape, E: error::Error> CalculateShape for Broadcast<($($A,)*)>
        where
            $($A: CalculateShape,)*
            ($($A::Shape,)*): BroadcastTogether<Output=O>,
            (<($($A::Shape,)*) as BroadcastTogether>::Error,): CommonError<Error=E>,
            $(E: From<<$A as CalculateShape>::Error>,)*
            E: From<<($($A::Shape,)*) as BroadcastTogether>::Error>,
        {
            type Shape = O;
            type Error = E;

            fn calculate_shape(&self) -> Result<Self::Shape, Self::Error> {
                let ($($a,)*) = &self.0;
                Ok(($(
                    $a.calculate_shape()?,
                )*).broadcast_together()?)
            }
        }

        impl<$($A,)* E: error::Error> Build for Broadcast<($($A,)*)>
        where
            Self: CalculateShape<Shape: Shape>,
            $($A: BuildWithShape<<Self as CalculateShape>::Shape>,)*
            (<Self as CalculateShape>::Error, $(<$A as BuildWithShape<<Self as CalculateShape>::Shape>>::Error,)*): CommonError<Error=E>,
            E: From<<Self as CalculateShape>::Error>,
            $(E: From<<$A as BuildWithShape<<Self as CalculateShape>::Shape>>::Error>,)*
        {
            type Output = ($(<$A as BuildWithShape<<Self as CalculateShape>::Shape>>::Output,)*);
            type Error = E;

            fn build(self) -> Result<Self::Output, Self::Error> {
                let shape = self.calculate_shape()?;
                let ($($a,)*) = self.0;
                Ok(($(
                    $a.build_with_shape(shape)?,
                )*))
            }
        }

        impl<$($A,)* S: Shape, E: error::Error> BuildWithShape<S> for Broadcast<($($A,)*)>
        where
            Self: CalculateShape,
            $($A: BuildWithShape<S>,)*
            ($(<$A as BuildWithShape<S>>::Error,)*): CommonError<Error=E>,
            $(E: From<<$A as BuildWithShape<S>>::Error>,)*
        {
            type Output = ($(<$A as BuildWithShape<S>>::Output,)*);
            type Error = E;

            fn build_with_shape(self, shape: S) -> Result<Self::Output, Self::Error> {
                let ($($a,)*) = self.0;
                Ok(($(
                    $a.build_with_shape(shape)?,
                )*))
            }
        }
    };
}

impl_broadcast_builder!();
impl_broadcast_builder!(a0 A0);
impl_broadcast_builder!(a0 A0 a1 A1);
impl_broadcast_builder!(a0 A0 a1 A1 a2 A2);
impl_broadcast_builder!(a0 A0 a1 A1 a2 A2 a3 A3);

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

/// The inetermediate representation of output data.
/// This generally serves two purposes:
/// * Providing a `.view_mut()` method to which data can be written
/// * Providing a `.output()` method to consume it and produce a return value
pub trait OutTarget {
    type Output;
    type Shape: Shape;
    type Data: ?Sized;

    fn view_mut(&mut self) -> ViewMut<Self::Shape, Self::Data>;

    /// Consume this output target and produce its canonical return value
    /// (e.g. `()` for [ViewMut] or [Array] for [ArrayTarget])
    fn output(self) -> Self::Output;
}

impl<'a, S: Shape, D: ?Sized> OutTarget for ViewMut<'a, S, D> {
    type Output = ();
    type Shape = S;
    type Data = D;

    fn view_mut(&mut self) -> ViewMut<Self::Shape, Self::Data> {
        ViewMut {
            shape: self.shape,
            offset: self.offset,
            strides: self.strides,
            data: self.data,
        }
    }

    fn output(self) -> Self::Output {
        ()
    }
}

pub struct AllocShape<S, E> {
    shape: S,
    element: marker::PhantomData<E>,
}

pub struct Alloc<E> {
    element: marker::PhantomData<E>,
}

pub fn alloc_shape<S: Shape, E>(shape: S) -> AllocShape<S, E> {
    AllocShape {
        shape,
        element: marker::PhantomData,
    }
}

pub fn alloc<E>() -> Alloc<E> {
    Alloc {
        element: marker::PhantomData,
    }
}

/// An output target wrapping an owned [Array],
/// that when viewed has a different (but equal) shape
/// than the underlying `Array`.
///
/// This should never need to be constructed directly
pub struct ArrayTarget<S: Shape, S2: Shape + BroadcastInto<S>, D> {
    array: Array<S2, D>,
    shape: S,
}

impl<
        'a,
        S: Shape,
        S2: BroadcastIntoConcrete<S> + BroadcastIntoNoAlias<S>,
        D: ?Sized,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > OutTarget for ArrayTarget<S, S2, D2>
{
    type Output = Array<S2, D2>;
    type Shape = S;
    type Data = D;

    fn view_mut(&mut self) -> ViewMut<S, D> {
        ViewMut {
            shape: self.shape,
            offset: self.array.offset,
            strides: S2::convert_index(self.array.strides),
            data: &mut self.array.data,
        }
    }

    fn output(self) -> Self::Output {
        self.array
    }
}

/*
impl<'a, S: Shape + BroadcastIntoNoAlias<S>, E: Default + Clone> IntoTarget for AllocShape<S, E> {
    type Shape = S;
    type Data = [E];

    fn shape(&self) -> Self::Shape {
        self.shape
    }
}
*/

impl<S: Shape, D: ?Sized, D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>> OutTarget
    for Array<S, D2>
{
    type Output = Self;
    type Shape = S;
    type Data = D;

    fn view_mut(&mut self) -> ViewMut<S, D> {
        ViewMut {
            shape: self.shape,
            offset: self.offset,
            strides: self.strides,
            data: &mut self.data,
        }
    }

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

pub trait CommonError {
    type Error;
}

impl CommonError for () {
    type Error = convert::Infallible;
}

macro_rules! impl_common_error {
    ($($a:ident $A:ident)*) => {
        impl<$($A,)*> CommonError for ($($A,)* convert::Infallible,)
            where ($($A,)*): CommonError
        {
            type Error = <($($A,)*) as CommonError>::Error;
        }

        impl<$($A,)*> CommonError for ($($A,)* BroadcastError,)
            where ($($A,)*): CommonError
        {
            type Error = BroadcastError;
        }
    };
}

impl_common_error!();
impl_common_error!(a0 A0);
impl_common_error!(a0 A0 a1 A1);
impl_common_error!(a0 A0 a1 A1 a2 A2);

/////////////////////////////////////////////

#[cfg(test)]
mod test {

    use crate::{
        alloc, alloc_shape, Array, AsIndex, Broadcast, BroadcastTogether, Build, Const,
        DefiniteRange, NewAxis, Out, OutTarget, Shape, View,
    };

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

        fn bernstein_coef<A, B, O: OutTarget<Shape = S, Data = [f32]>, S: Shape, E>(
            c_m: &A,
            out: B,
        ) -> Result<O::Output, E>
        where
            for<'a> Broadcast<(&'a A, Out<B>)>: Build<Output = (View<'a, S, [f32]>, O), Error = E>,
        {
            let (c_m, mut out_target) = Broadcast((c_m, Out(out))).build()?;
            let mut out = out_target.view_mut();

            // XXX
            //let val1 = c_m[<<V as IntoView>::Shape as Shape>::Index::zero()];

            for (i, out_entry) in (&mut out).into_iter().nd_enumerate() {
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

            Ok(out_target.output())
        }

        // TEST DATA

        let a = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
            offset: 0,
            data: vec![1., 2., 3., 4.],
        };

        let mut b = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (2, 2).default_strides(),
            offset: 0,
            data: vec![0.; 4],
        };

        let Ok(_) = bernstein_coef(&a, &mut b);
        let Ok(_) = bernstein_coef(&a, alloc());
        let Ok(_) = bernstein_coef(&a, alloc_shape((Const::<2>, Const::<2>)));
        bernstein_coef(&a, alloc_shape((2, 2))).unwrap();
    }

    #[test]
    fn test_sum_prod() {
        fn sum_prod<A, B, S: Shape, E>(in1: &A, in2: &B) -> Result<f32, E>
        where
            for<'a> Broadcast<(&'a A, &'a B)>:
                Build<Output = (View<'a, S, [f32]>, View<'a, S, [f32]>), Error = E>,
        {
            let (in1, in2) = Broadcast((in1, in2)).build()?;

            Ok(in1
                .into_iter()
                .zip(in2.into_iter())
                .map(|(a, b)| a * b)
                .sum())
        }

        let a = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
            offset: 0,
            data: vec![1., 2., 3., 4.],
        };

        sum_prod(&a, &a).unwrap();
    }

    #[test]
    fn test_add() {
        fn add<A, B, C, O: OutTarget<Shape = S, Data = [f32]>, S: Shape, E>(
            a: &A,
            b: &B,
            out: C,
        ) -> Result<O::Output, E>
        where
            for<'a> Broadcast<(&'a A, &'a B, Out<C>)>:
                Build<Output = (View<'a, S, [f32]>, View<'a, S, [f32]>, O), Error = E>,
        {
            let (a, b, mut out_target) = Broadcast((a, b, Out(out))).build()?;
            let mut out = out_target.view_mut();

            for (out, (a, b)) in (&mut out).into_iter().zip(a.into_iter().zip(b.into_iter())) {
                *out = a + b;
            }

            Ok(out_target.output())
        }

        let a = Array {
            shape: (Const::<2>,),
            strides: (Const::<2>,).default_strides(),
            offset: 0,
            data: vec![1., 2.],
        };

        let b = Array {
            shape: (Const::<2>, NewAxis),
            strides: (Const::<2>, NewAxis).default_strides(),
            offset: 0,
            data: vec![10., 20.],
        };

        let mut c = Array {
            shape: (Const::<2>, Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>, Const::<2>).default_strides(),
            offset: 0,
            data: vec![0.; 8],
        };

        add(&a, &b, &mut c).unwrap();

        assert_eq!(c.data, [11., 12., 21., 22., 11., 12., 21., 22.]);
    }

    #[test]
    fn test_sum() {
        fn sum<A, S: Shape>(a: &A) -> f32
        where
            for<'a> &'a A: Build<Output = View<'a, S, [f32]>>,
        {
            let a = a.build().unwrap(); // Infallible?

            a.into_iter().sum()
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
        fn ones<A, O: OutTarget<Data = [f32]>>(out: A) -> O::Output
        where
            Out<A>: Build<Output = O>,
        {
            let mut out_target = Out(out).build().unwrap(); // Infallible?
            let mut out = out_target.view_mut();

            for e in &mut out {
                *e = 1.;
            }

            out_target.output()
        }

        let mut a = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
            offset: 0,
            data: vec![1., 2., 3., 4.],
        };

        ones(&mut a);
        ones(alloc_shape((4, 4)));
    }

    #[test]
    fn test_broadcast() {
        let Ok(s) = ((Const::<3>, Const::<4>), (Const::<3>, Const::<4>)).broadcast_together();
        assert_eq!(s.as_index(), [3, 4]);
        assert_eq!(
            ((Const::<3>, Const::<4>), (Const::<3>, 4))
                .broadcast_together()
                .unwrap()
                .as_index(),
            [3, 4]
        );
        assert_eq!(
            ((Const::<3>, Const::<4>), (3, 4))
                .broadcast_together()
                .unwrap()
                .as_index(),
            [3, 4]
        );
        assert!(((Const::<3>, Const::<4>), (Const::<3>, 5))
            .broadcast_together()
            .is_err());
        assert!(((Const::<3>, Const::<4>), (3, 5))
            .broadcast_together()
            .is_err());

        let Ok(s) = ((Const::<3>, Const::<4>), ()).broadcast_together();
        assert_eq!(s.as_index(), [3, 4]);
        assert_eq!(
            ((Const::<3>, Const::<4>), (4,))
                .broadcast_together()
                .unwrap()
                .as_index(),
            [3, 4]
        );
        assert_eq!(
            ((Const::<3>, Const::<4>), (3, NewAxis))
                .broadcast_together()
                .unwrap()
                .as_index(),
            [3, 4]
        );
        assert_eq!(
            ((Const::<3>, Const::<4>), (10, 3, NewAxis))
                .broadcast_together()
                .unwrap()
                .as_index(),
            [10, 3, 4]
        );
    }
}
