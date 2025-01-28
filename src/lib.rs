use std::fmt;
use std::marker;
use std::ops;

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

pub trait BroadcastInto<T> {
    fn can_broadcast_into(self, other: T) -> bool;
}

pub trait IntoIndex<T: Shape>: Shape + BroadcastInto<T> {
    fn into_index(index: Self::Index) -> T::Index;
}

pub trait BroadcastIntoNoAlias<T>: BroadcastInto<T> {}

pub trait BroadcastWith<T: BroadcastInto<Self::Output>>: BroadcastInto<Self::Output> {
    type Output;

    fn broadcast_with(self, other: T) -> Option<Self::Output>;
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
/// **no two shapes should be assumed to be the same type**.
/// // TODO explain how to write a "same shape" type bound
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
        Self: BroadcastWith<S>,
    {
        self.broadcast_with(other).unwrap_or_else(|| {
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

pub trait Push {
    type OneBigger: Pop<OneSmaller = Self>;

    fn append(self, a: usize) -> Self::OneBigger;
}

pub trait Pop {
    type OneSmaller: Push<OneBigger = Self>;

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

pub trait TuplePush {
    type OneBigger<A>: TuplePop<Head = A, OneSmaller = Self>;

    fn append<A>(self, a: A) -> Self::OneBigger<A>;
}

pub trait TuplePop {
    type Head;
    type OneSmaller: TuplePush<OneBigger<Self::Head> = Self>;

    fn split(self) -> (Self::OneSmaller, Self::Head);
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
            type Head = A;
            type OneSmaller = ($($A,)*);

            fn split(self) -> (Self::OneSmaller, Self::Head) {
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
            where ($($A,)* Const<N>,): 'static + Copy + fmt::Debug + AsIndex
        {}

        impl<$($A,)*> Shape for ($($A,)* usize,)
            where ($($A,)* usize,): 'static + Copy + fmt::Debug + AsIndex
        {}

        impl<$($A,)*> Shape for ($($A,)* NewAxis,)
            where ($($A,)* NewAxis,): 'static + Copy + fmt::Debug + AsIndex
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

impl BroadcastInto<()> for () {
    fn can_broadcast_into(self, _other: ()) -> bool {
        true
    }
}

impl IntoIndex<()> for () {
    fn into_index(_index: [usize; 0]) -> [usize; 0] {
        []
    }
}
impl BroadcastIntoNoAlias<()> for () {}

macro_rules! impl_broadcast_into_a {
    ($($a:ident $A:ident)*) => {
        impl<$($A: Copy + fmt::Debug,)*> BroadcastInto<()> for ($($A,)* NewAxis,)
            where ($($A,)*): BroadcastInto<()>,
        {
            fn can_broadcast_into(self, _other: ()) -> bool {
                let ($($a,)* _,) = self;
                ($($a,)*).can_broadcast_into(())
            }
        }

        impl<$($A: Copy + fmt::Debug,)*> BroadcastIntoNoAlias<()> for ($($A,)* NewAxis,)
            where ($($A,)*): IntoIndex<()>,
            ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
        {}

        impl<$($A: Copy + fmt::Debug,)*> IntoIndex<()> for ($($A,)* NewAxis,)
            where ($($A,)*): IntoIndex<()>,
            ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
        {
            fn into_index(index: <($($A,)*) as AsIndex>::Index) -> <() as AsIndex>::Index {
                <($($A,)*) as IntoIndex<()>>::into_index(index)
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
            fn can_broadcast_into(self, other: ($($B,)* NewAxis,)) -> bool {
                let ($($b,)* _,) = other;
                ().can_broadcast_into(($($b,)*))
            }
        }
        impl<$($B,)*> BroadcastIntoNoAlias<($($B,)* NewAxis,)> for ()
            where (): BroadcastIntoNoAlias<($($B,)*)>,
            ($($B,)*): Shape,
            ($($B,)* NewAxis,): Shape<Index=<($($B,)*) as AsIndex>::Index>,
        {}

        impl<$($B,)*> IntoIndex<($($B,)* NewAxis,)> for ()
            where (): IntoIndex<($($B,)*)>,
            ($($B,)*): Shape,
            ($($B,)* NewAxis,): Shape<Index=<($($B,)*) as AsIndex>::Index>,
        {
            fn into_index(index: [usize; 0]) -> <($($B,)*) as AsIndex>::Index {
                <() as IntoIndex<($($B,)*)>>::into_index(index)
            }
        }

        impl<const N: usize, $($B,)*> BroadcastInto<($($B,)* Const<N>,)> for ()
            where (): BroadcastInto<($($B,)*)>,
            ($($B,)*): Shape<Index: Push<OneBigger=<($($B,)* Const<N>,) as AsIndex>::Index>>,
            ($($B,)* Const<N>,): Shape,
        {
            fn can_broadcast_into(self, other: ($($B,)* Const<N>,)) -> bool {
                let ($($b,)* _,) = other;
                ().can_broadcast_into(($($b,)*))
            }
        }
        //impl<const N: usize> BroadcastIntoNoAlias<(S2, Const<N>)> for ()
        //{ ! }
        impl<const N: usize, $($B,)*> IntoIndex<($($B,)* Const<N>,)> for ()
            where (): IntoIndex<($($B,)*)>,
            ($($B,)*): Shape<Index: Push<OneBigger=<($($B,)* Const<N>,) as AsIndex>::Index>>,
            ($($B,)* Const<N>,): Shape,
        {
            fn into_index(index: [usize; 0]) -> <($($B,)* Const<N>,) as AsIndex>::Index {
                <() as IntoIndex<($($B,)*)>>::into_index(index).append(0)
            }
        }

        impl<$($B,)*> BroadcastInto<($($B,)* usize,)> for ()
            where (): BroadcastInto<($($B,)*)>,
            ($($B,)*): Shape<Index: Push<OneBigger=<($($B,)* usize,) as AsIndex>::Index>>,
            ($($B,)* usize,): Shape,
        {
            fn can_broadcast_into(self, other: ($($B,)* usize,)) -> bool {
                let ($($b,)* _,) = other;
                ().can_broadcast_into(($($b,)*))
            }
        }
        //impl BroadcastIntoNoAlias<(S2, usize)> for ()
        //{ ! }
        impl<$($B,)*> IntoIndex<($($B,)* usize,)> for ()
            where (): IntoIndex<($($B,)*)>,
            ($($B,)*): Shape<Index: Push<OneBigger=<($($B,)* usize,) as AsIndex>::Index>>,
            ($($B,)* usize,): Shape,
        {
            fn into_index(index: [usize; 0]) -> <($($B,)* usize,) as AsIndex>::Index {
                <() as IntoIndex<($($B,)*)>>::into_index(index).append(0)
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
            where ($($A,)*): IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* NewAxis,): Shape<Index=<($($B,)*) as AsIndex>::Index>,
        {
            fn can_broadcast_into(self, other: ($($B,)* NewAxis,)) -> bool {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                ($($a,)*).can_broadcast_into(($($b,)*))
            }

        }
        impl<$($A,)* $($B,)*> BroadcastIntoNoAlias<($($B,)* NewAxis,)> for ($($A,)* NewAxis,)
            where ($($A,)*): IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* NewAxis,): Shape<Index=<($($B,)*) as AsIndex>::Index>,
        {}
        impl<$($A,)* $($B,)*> IntoIndex<($($B,)* NewAxis,)> for ($($A,)* NewAxis,)
            where ($($A,)*): IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* NewAxis,): Shape<Index=<($($B,)*) as AsIndex>::Index>,
        {
            fn into_index(index: <($($A,)* NewAxis,) as AsIndex>::Index) -> <($($B,)*) as AsIndex>::Index {
                <($($A,)*) as IntoIndex<($($B,)*)>>::into_index(index)
            }
        }

        impl<const N: usize, $($A,)* $($B,)*> BroadcastInto<($($B,)* Const<N>,)> for ($($A,)* NewAxis,)
            where ($($A,)*): IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn can_broadcast_into(self, other: ($($B,)* Const<N>,)) -> bool {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                ($($a,)*).can_broadcast_into(($($b,)*))
            }
        }
        //impl<const N: usize, $($A,)* $($B,)*> BroadcastInto<($($B,)* Const<N>,)> for ($($A,)* NewAxis,) { ! }
        impl<const N: usize, $($A,)* $($B,)*> IntoIndex<($($B,)* Const<N>,)> for ($($A,)* NewAxis,)
            where ($($A,)*): IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn into_index(index: <($($A,)* NewAxis,) as AsIndex>::Index) -> <($($B,)* Const<N>,) as AsIndex>::Index {
                <($($A,)*) as IntoIndex<($($B,)*)>>::into_index(index).append(0)
            }
        }

        impl<$($A,)* $($B,)*> BroadcastInto<($($B,)* usize,)> for ($($A,)* NewAxis,)
            where ($($A,)*): IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn can_broadcast_into(self, other: ($($B,)* usize,)) -> bool {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                ($($a,)*).can_broadcast_into(($($b,)*))
            }
        }
        //impl<$($A,)* $($B,)*> BroadcastInto<($($B,)* Const<N>,)> for ($($A,)* NewAxis,) { ! }
        impl<$($A,)* $($B,)*> IntoIndex<($($B,)* usize,)> for ($($A,)* NewAxis,)
            where ($($A,)*): IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* NewAxis,): Shape<Index=<($($A,)*) as AsIndex>::Index>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn into_index(index: <($($A,)* NewAxis,) as AsIndex>::Index) -> <($($B,)* usize,) as AsIndex>::Index {
                <($($A,)*) as IntoIndex<($($B,)*)>>::into_index(index).append(0)
            }
        }

        impl<const N: usize, $($A,)* $($B,)*> BroadcastInto<($($B,)* Const<N>,)> for ($($A,)* Const<N>,)
            where ($($A,)*): Shape<Index: Push> + IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* Const<N>,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn can_broadcast_into(self, other: ($($B,)* Const<N>,)) -> bool {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                ($($a,)*).can_broadcast_into(($($b,)*))
            }
        }
        impl<const N: usize, $($A,)* $($B,)*> BroadcastIntoNoAlias<($($B,)* Const<N>,)> for ($($A,)* Const<N>,)
            where ($($A,)*): Shape<Index: Push> + IntoIndex<($($B,)*)> + BroadcastIntoNoAlias<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* Const<N>,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {}
        impl<const N: usize, $($A,)* $($B,)*> IntoIndex<($($B,)* Const<N>,)> for ($($A,)* Const<N>,)
            where ($($A,)*): Shape<Index: Push> + IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* Const<N>,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn into_index(index: <($($A,)* Const<N>,) as AsIndex>::Index) -> <($($B,)* Const<N>,) as AsIndex>::Index {
                let (index_rest, index_last) = index.split();
                <($($A,)*) as IntoIndex<($($B,)*)>>::into_index(index_rest).append(index_last)
            }
        }

        impl<const N: usize, $($A,)* $($B,)*> BroadcastInto<($($B,)* usize,)> for ($($A,)* Const<N>,)
            where ($($A,)*): Shape<Index: Push> + IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* Const<N>,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn can_broadcast_into(self, other: ($($B,)* usize,)) -> bool {
                let ($($a,)* _,) = self;
                let ($($b,)* b,) = other;
                ($($a,)*).can_broadcast_into(($($b,)*)) && N == b
            }
        }
        impl<const N: usize, $($A,)* $($B,)*> BroadcastIntoNoAlias<($($B,)* usize,)> for ($($A,)* Const<N>,)
            where ($($A,)*): Shape<Index: Push> + IntoIndex<($($B,)*)> + BroadcastIntoNoAlias<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* Const<N>,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {}
        impl<const N: usize, $($A,)* $($B,)*> IntoIndex<($($B,)* usize,)> for ($($A,)* Const<N>,)
            where ($($A,)*): Shape<Index: Push> + IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* Const<N>,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn into_index(index: <($($A,)* Const<N>,) as AsIndex>::Index) -> <($($B,)* usize,) as AsIndex>::Index {
                let (index_rest, index_last) = index.split();
                <($($A,)*) as IntoIndex<($($B,)*)>>::into_index(index_rest).append(index_last)
            }
        }

        impl<const N: usize, $($A,)* $($B,)*> BroadcastInto<($($B,)* Const<N>,)> for ($($A,)* usize,)
            where ($($A,)*): Shape<Index: Push> + IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* usize,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn can_broadcast_into(self, other: ($($B,)* Const<N>,)) -> bool {
                let ($($a,)* a,) = self;
                let ($($b,)* _,) = other;
                ($($a,)*).can_broadcast_into(($($b,)*)) && a == N
            }
        }
        impl<const N: usize, $($A,)* $($B,)*> BroadcastIntoNoAlias<($($B,)* Const<N>,)> for ($($A,)* usize,)
            where ($($A,)*): Shape<Index: Push> + IntoIndex<($($B,)*)> + BroadcastIntoNoAlias<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* usize,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {}
        impl<const N: usize, $($A,)* $($B,)*> IntoIndex<($($B,)* Const<N>,)> for ($($A,)* usize,)
            where ($($A,)*): Shape<Index: Push> + IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* usize,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* Const<N>,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn into_index(index: <($($A,)* usize,) as AsIndex>::Index) -> <($($B,)* Const<N>,) as AsIndex>::Index {
                let (index_rest, index_last) = index.split();
                <($($A,)*) as IntoIndex<($($B,)*)>>::into_index(index_rest).append(index_last)
            }
        }

        impl<$($A,)* $($B,)*> BroadcastInto<($($B,)* usize,)> for ($($A,)* usize,)
            where ($($A,)*): Shape<Index: Push> + IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* usize,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn can_broadcast_into(self, other: ($($B,)* usize,)) -> bool {
                let ($($a,)* a,) = self;
                let ($($b,)* b,) = other;
                ($($a,)*).can_broadcast_into(($($b,)*)) && a == b
            }
        }
        impl<$($A,)* $($B,)*> BroadcastIntoNoAlias<($($B,)* usize,)> for ($($A,)* usize,)
            where ($($A,)*): Shape<Index: Push> + IntoIndex<($($B,)*)> + BroadcastIntoNoAlias<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* usize,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {}
        impl<$($A,)* $($B,)*> IntoIndex<($($B,)* usize,)> for ($($A,)* usize,)
            where ($($A,)*): Shape<Index: Push> + IntoIndex<($($B,)*)>,
                  ($($B,)*): Shape<Index: Push>,
                  ($($A,)* usize,): Shape<Index=<<($($A,)*) as AsIndex>::Index as Push>::OneBigger>,
                  ($($B,)* usize,): Shape<Index=<<($($B,)*) as AsIndex>::Index as Push>::OneBigger>,
        {
            fn into_index(index: <($($A,)* usize,) as AsIndex>::Index) -> <($($B,)* usize,) as AsIndex>::Index {
                let (index_rest, index_last) = index.split();
                <($($A,)*) as IntoIndex<($($B,)*)>>::into_index(index_rest).append(index_last)
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

#[derive(Clone, Copy, Debug)]
pub struct AnyShape;

impl BroadcastInto<AnyShape> for AnyShape {
    fn can_broadcast_into(self, _other: AnyShape) -> bool {
        true
    }
}

impl<S: Shape> BroadcastInto<S> for AnyShape {
    fn can_broadcast_into(self, _other: S) -> bool {
        true
    }
}

// BroadcastWith

macro_rules! impl_broadcast_with_a {
    () => {
        impl BroadcastWith<()> for () {
            type Output = ();

            fn broadcast_with(self, _other: ()) -> Option<Self::Output> {
                Some(())
            }
        }
        impl BroadcastWith<AnyShape> for AnyShape {
            type Output = AnyShape;

            fn broadcast_with(self, _other: AnyShape) -> Option<Self::Output> {
                Some(AnyShape)
            }
        }
        impl BroadcastWith<()> for AnyShape {
            type Output = ();

            fn broadcast_with(self, _other: ()) -> Option<Self::Output> {
                Some(())
            }
        }
        impl BroadcastWith<AnyShape> for () {
            type Output = ();

            fn broadcast_with(self, _other: AnyShape) -> Option<Self::Output> {
                Some(())
            }
        }
    };

    ($($a:ident $A:ident)*) => {
        impl<$($A,)*> BroadcastWith<()> for ($($A,)*)
            where (): BroadcastInto<($($A,)*)>,
                  ($($A,)*): BroadcastInto<($($A,)*)>,
        {
            type Output = ($($A,)*);

            fn broadcast_with(self, _other: ()) -> Option<Self::Output> {
                Some(self)
            }
        }
        impl<$($A,)*> BroadcastWith<AnyShape> for ($($A,)*)
            where AnyShape: BroadcastInto<($($A,)*)>,
                  ($($A,)*): BroadcastInto<($($A,)*)>,
        {
            type Output = ($($A,)*);

            fn broadcast_with(self, _other: AnyShape) -> Option<Self::Output> {
                Some(self)
            }
        }


        impl<$($A,)*> BroadcastWith<($($A,)*)> for ()
            where (): BroadcastInto<($($A,)*)>,
                  ($($A,)*): BroadcastInto<($($A,)*)>,
        {
            type Output = ($($A,)*);

            fn broadcast_with(self, other: ($($A,)*)) -> Option<Self::Output> {
                Some(other)
            }
        }
        impl<$($A,)*> BroadcastWith<($($A,)*)> for AnyShape
            where AnyShape: BroadcastInto<($($A,)*)>,
                  ($($A,)*): BroadcastInto<($($A,)*)>,
        {
            type Output = ($($A,)*);

            fn broadcast_with(self, other: ($($A,)*)) -> Option<Self::Output> {
                Some(other)
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

            fn broadcast_with(self, other: ($($B,)* NewAxis,)) -> Option<Self::Output> {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                Some(($($a,)*).broadcast_with(($($b,)*))?.append(NewAxis))
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

            fn broadcast_with(self, other: ($($B,)* Const<N>,)) -> Option<Self::Output> {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                Some(($($a,)*).broadcast_with(($($b,)*))?.append(Const))
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

            fn broadcast_with(self, other: ($($B,)* usize,)) -> Option<Self::Output> {
                let ($($a,)* _,) = self;
                let ($($b,)* b,) = other;
                Some(($($a,)*).broadcast_with(($($b,)*))?.append(b))
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

            fn broadcast_with(self, other: ($($B,)* NewAxis,)) -> Option<Self::Output> {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                Some(($($a,)*).broadcast_with(($($b,)*))?.append(Const))
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

            fn broadcast_with(self, other: ($($B,)* Const<N>,)) -> Option<Self::Output> {
                let ($($a,)* _,) = self;
                let ($($b,)* _,) = other;
                Some(($($a,)*).broadcast_with(($($b,)*))?.append(Const))
            }
        }

        impl<const N: usize, $($A,)* $($B,)* O,> BroadcastWith<($($B,)* usize,)> for ($($A,)* Const<N>,)
            where O: Shape,
                  ($($A,)*): BroadcastWith<($($B,)*), Output: TuplePush<OneBigger<Const<N>>=O>>,
                  ($($B,)*): BroadcastInto<<($($A,)*) as BroadcastWith<($($B,)*)>>::Output>,
                  ($($A,)* Const<N>,): BroadcastInto<O>,
                  ($($B,)* usize,): BroadcastInto<O>,
        {
            type Output = O;

            fn broadcast_with(self, other: ($($B,)* usize,)) -> Option<Self::Output> {
                let ($($a,)* _,) = self;
                let ($($b,)* b,) = other;
                if N != b {
                    return None;
                }
                Some(($($a,)*).broadcast_with(($($b,)*))?.append(Const))
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

            fn broadcast_with(self, other: ($($B,)* NewAxis,)) -> Option<Self::Output> {
                let ($($a,)* a,) = self;
                let ($($b,)* _,) = other;
                Some(($($a,)*).broadcast_with(($($b,)*))?.append(a))
            }
        }

        impl<const N: usize, $($A,)* $($B,)* O,> BroadcastWith<($($B,)* Const<N>,)> for ($($A,)* usize,)
            where O: Shape,
                  ($($A,)*): BroadcastWith<($($B,)*), Output: TuplePush<OneBigger<Const<N>>=O>>,
                  ($($B,)*): BroadcastInto<<($($A,)*) as BroadcastWith<($($B,)*)>>::Output>,
                  ($($A,)* usize,): BroadcastInto<O>,
                  ($($B,)* Const<N>,): BroadcastInto<O>,
        {
            type Output = O;

            fn broadcast_with(self, other: ($($B,)* Const<N>,)) -> Option<Self::Output> {
                let ($($a,)* a,) = self;
                let ($($b,)* _,) = other;
                if a != N {
                    return None;
                }
                Some(($($a,)*).broadcast_with(($($b,)*))?.append(Const))
            }
        }

        impl<$($A,)* $($B,)* O,> BroadcastWith<($($B,)* usize,)> for ($($A,)* usize,)
            where O: Shape,
                  ($($A,)*): BroadcastWith<($($B,)*), Output: TuplePush<OneBigger<usize>=O>>,
                  ($($B,)*): BroadcastInto<<($($A,)*) as BroadcastWith<($($B,)*)>>::Output>,
                  ($($A,)* usize,): BroadcastInto<O>,
                  ($($B,)* usize,): BroadcastInto<O>,
        {
            type Output = O;

            fn broadcast_with(self, other: ($($B,)* usize,)) -> Option<Self::Output> {
                let ($($a,)* a,) = self;
                let ($($b,)* b,) = other;
                if a != b {
                    return None;
                }
                Some(($($a,)*).broadcast_with(($($b,)*))?.append(a))
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

pub trait BroadcastTogether {
    type Output;

    fn broadcast_together(self) -> Option<Self::Output>;
}

impl BroadcastTogether for () {
    type Output = ();

    fn broadcast_together(self) -> Option<Self::Output> {
        Some(())
    }
}

macro_rules! impl_broadcast_together {
    ($($a:ident $A:ident)*) => {
        impl<$($A,)* A, O> BroadcastTogether for ($($A,)* A,)
        where
            ($($A,)*): BroadcastTogether<Output=O>,
            O: BroadcastWith<A>,
            A: BroadcastInto<<O as BroadcastWith<A>>::Output>,
        {
            type Output = <O as BroadcastWith<A>>::Output;

            fn broadcast_together(self) -> Option<Self::Output> {
                let ($($a,)* a,) = self;
                Some(($($a,)*).broadcast_together()?.broadcast_with(a)?)
            }
        }
    };
}

impl_broadcast_together!();
impl_broadcast_together!(a0 A0);
impl_broadcast_together!(a0 A0 a1 A1);
impl_broadcast_together!(a0 A0 a1 A1 a2 A2);

pub trait Build {
    type Output;

    fn build(self) -> Option<Self::Output>;
}

pub trait BuildWithShape<S: Shape> {
    type Output;

    fn build_with_shape(self, shape: S) -> Option<Self::Output>;
}

pub trait Shaped {
    type Shape;

    fn shape(&self) -> Option<Self::Shape>;
}

impl<S: Shape, D: ?Sized, D2: ops::Deref<Target = D>> Shaped for &Array<S, D2> {
    type Shape = S;

    fn shape(&self) -> Option<Self::Shape> {
        Some(self.shape)
    }
}

impl<S: Shape, D: ?Sized> Shaped for &View<'_, S, D> {
    type Shape = S;

    fn shape(&self) -> Option<Self::Shape> {
        Some(self.shape)
    }
}

impl<S: Shape, D: ?Sized> Shaped for &ViewMut<'_, S, D> {
    type Shape = S;

    fn shape(&self) -> Option<Self::Shape> {
        Some(self.shape)
    }
}

impl<'a, S: Shape, D: ?Sized + 'a, D2: ops::Deref<Target = D>> Build for &'a Array<S, D2> {
    type Output = View<'a, S, D>;

    fn build(self) -> Option<View<'a, S, D>> {
        Some(View {
            shape: self.shape,
            offset: self.offset,
            strides: self.strides,
            data: &self.data,
        })
    }
}

impl<'a, S: Shape, S2: IntoIndex<S>, D: ?Sized + 'a, D2: ops::Deref<Target = D>> BuildWithShape<S>
    for &'a Array<S2, D2>
{
    type Output = View<'a, S, D>;

    fn build_with_shape(self, shape: S) -> Option<Self::Output> {
        if !self.shape.can_broadcast_into(shape) {
            return None;
        }
        Some(View {
            shape,
            offset: self.offset,
            strides: S2::into_index(self.strides),
            data: &self.data,
        })
    }
}

impl<'a, S: Shape, D: ?Sized> Build for &'a View<'a, S, D> {
    type Output = View<'a, S, D>;

    fn build(self) -> Option<View<'a, S, D>> {
        Some(View {
            shape: self.shape,
            offset: self.offset,
            strides: self.strides,
            data: self.data,
        })
    }
}

impl<'a, S: Shape, S2: IntoIndex<S>, D: ?Sized> BuildWithShape<S> for &'a View<'a, S2, D> {
    type Output = View<'a, S, D>;

    fn build_with_shape(self, shape: S) -> Option<Self::Output> {
        if !self.shape.can_broadcast_into(shape) {
            return None;
        }
        Some(View {
            shape,
            offset: self.offset,
            strides: S2::into_index(self.strides),
            data: self.data,
        })
    }
}

impl<'a, S: Shape, D: ?Sized> Build for &'a ViewMut<'a, S, D> {
    type Output = View<'a, S, D>;

    fn build(self) -> Option<View<'a, S, D>> {
        Some(View {
            shape: self.shape,
            offset: self.offset,
            strides: self.strides,
            data: self.data,
        })
    }
}

impl<'a, S: Shape, S2: IntoIndex<S>, D: ?Sized> BuildWithShape<S> for &'a ViewMut<'a, S2, D> {
    type Output = View<'a, S, D>;

    fn build_with_shape(self, shape: S) -> Option<Self::Output> {
        if !self.shape.can_broadcast_into(shape) {
            return None;
        }
        Some(View {
            shape,
            offset: self.offset,
            strides: S2::into_index(self.strides),
            data: self.data,
        })
    }
}

impl<'a, S: Shape, D: ?Sized + 'a, D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>> Build
    for &'a mut Array<S, D2>
{
    type Output = ViewMut<'a, S, D>;

    fn build(self) -> Option<ViewMut<'a, S, D>> {
        Some(ViewMut {
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
        S2: IntoIndex<S> + BroadcastIntoNoAlias<S>,
        D: ?Sized + 'a,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > BuildWithShape<S> for &'a mut Array<S2, D2>
{
    type Output = ViewMut<'a, S, D>;

    fn build_with_shape(self, shape: S) -> Option<Self::Output> {
        if !self.shape.can_broadcast_into(shape) {
            return None;
        }
        Some(ViewMut {
            shape,
            offset: self.offset,
            strides: S2::into_index(self.strides),
            data: &mut self.data,
        })
    }
}

impl<'a, S: Shape, D: ?Sized> Build for &'a mut ViewMut<'a, S, D> {
    type Output = ViewMut<'a, S, D>;

    fn build(self) -> Option<Self::Output> {
        Some(ViewMut {
            shape: self.shape,
            offset: self.offset,
            strides: self.strides,
            data: self.data,
        })
    }
}

impl<'a, S: Shape, S2: IntoIndex<S> + BroadcastIntoNoAlias<S>, D: ?Sized> BuildWithShape<S>
    for &'a mut ViewMut<'a, S2, D>
{
    type Output = ViewMut<'a, S, D>;

    fn build_with_shape(self, shape: S) -> Option<Self::Output> {
        if !self.shape.can_broadcast_into(shape) {
            return None;
        }
        Some(ViewMut {
            shape,
            offset: self.offset,
            strides: S2::into_index(self.strides),
            data: self.data,
        })
    }
}

struct OutMarker<T>(T);

impl<'a, S: Shape, D: 'a + ?Sized, D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>> Shaped
    for OutMarker<&'a mut Array<S, D2>>
{
    type Shape = S;

    fn shape(&self) -> Option<Self::Shape> {
        Some(self.0.shape)
    }
}

impl<'a, S: Shape, D: 'a + ?Sized, D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>> Build
    for OutMarker<&'a mut Array<S, D2>>
{
    type Output = ViewMut<'a, S, D>;

    fn build(self) -> Option<Self::Output> {
        self.0.build()
    }
}

impl<
        'a,
        S: Shape,
        S2: IntoIndex<S> + BroadcastIntoNoAlias<S>,
        D: 'a + ?Sized,
        D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>,
    > BuildWithShape<S> for OutMarker<&'a mut Array<S2, D2>>
{
    type Output = ViewMut<'a, S, D>;

    fn build_with_shape(self, shape: S) -> Option<Self::Output> {
        self.0.build_with_shape(shape)
    }
}

impl<'a, S: Shape, D: 'a + ?Sized> Shaped for OutMarker<&'a mut ViewMut<'a, S, D>> {
    type Shape = S;

    fn shape(&self) -> Option<Self::Shape> {
        Some(self.0.shape)
    }
}

impl<'a, S: Shape, D: 'a + ?Sized> Build for OutMarker<&'a mut ViewMut<'a, S, D>> {
    type Output = ViewMut<'a, S, D>;

    fn build(self) -> Option<Self::Output> {
        self.0.build()
    }
}

impl<'a, S: Shape, S2: IntoIndex<S> + BroadcastIntoNoAlias<S>, D: 'a + ?Sized> BuildWithShape<S>
    for OutMarker<&'a mut ViewMut<'a, S2, D>>
{
    type Output = ViewMut<'a, S, D>;

    fn build_with_shape(self, shape: S) -> Option<Self::Output> {
        self.0.build_with_shape(shape)
    }
}

impl<'a, S: Shape, E: Default + Clone> Shaped for OutMarker<AllocShape<S, E>> {
    type Shape = S;

    fn shape(&self) -> Option<Self::Shape> {
        Some(self.0.shape)
    }
}

impl<'a, E: Default + Clone> Shaped for OutMarker<Alloc<E>> {
    type Shape = AnyShape;

    fn shape(&self) -> Option<Self::Shape> {
        Some(AnyShape)
    }
}

impl<'a, S: Shape, E: Default + Clone> Build for OutMarker<AllocShape<S, E>> {
    type Output = Array<S, Vec<E>>;

    fn build(self) -> Option<Self::Output> {
        let OutMarker(AllocShape { shape, .. }) = self;
        Some(Array {
            shape,
            strides: shape.default_strides(),
            offset: 0,
            data: vec![E::default(); shape.num_elements()],
        })
    }
}

// Can't Build for Alloc since no shape information is available

impl<'a, S: Shape, S2: Shape + BroadcastIntoNoAlias<S>, E: Default + Clone> BuildWithShape<S>
    for OutMarker<AllocShape<S2, E>>
{
    type Output = ArrayTarget<S, S2, Vec<E>>;

    fn build_with_shape(self, shape: S) -> Option<Self::Output> {
        let OutMarker(AllocShape {
            shape: self_shape, ..
        }) = self;
        if !self_shape.can_broadcast_into(shape) {
            return None;
        }
        Some(ArrayTarget {
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

impl<'a, S: Shape, E: Default + Clone> BuildWithShape<S> for OutMarker<Alloc<E>> {
    type Output = Array<S, Vec<E>>;

    fn build_with_shape(self, shape: S) -> Option<Self::Output> {
        Some(Array {
            shape,
            strides: shape.default_strides(),
            offset: 0,
            data: vec![E::default(); shape.num_elements()],
        })
    }
}

pub struct Broadcast<T>(T);

macro_rules! impl_broadcast_builder {
    () => {
        impl Build for Broadcast<()>
        {
            type Output = ();

            fn build(self) -> Option<Self::Output> {
                Some(())
            }
        }

        impl<S: Shape> BuildWithShape<S> for Broadcast<()>
            where (): BroadcastInto<S>
        {
            type Output = ();

            fn build_with_shape(self, shape: S) -> Option<Self::Output> {
                if !().can_broadcast_into(shape) {
                    return None;
                }
                Some(())
            }
        }
    };

    ($($a:ident $A:ident)*) => {
        impl<$($A,)* O: Shape> Shaped for Broadcast<($($A,)*)>
        where
            $($A: Shaped,)*
            ($($A::Shape,)*): BroadcastTogether<Output=O>,
        {
            type Shape = O;

            fn shape(&self) -> Option<Self::Shape> {
                let ($($a,)*) = &self.0;
                ($(
                    $a.shape()?,
                )*).broadcast_together()
            }
        }

        impl<$($A,)*> Build for Broadcast<($($A,)*)>
        where
            Self: Shaped<Shape: Shape>,
            $($A: BuildWithShape<<Self as Shaped>::Shape>,)*
        {
            type Output = ($(<$A as BuildWithShape<<Self as Shaped>::Shape>>::Output,)*);

            fn build(self) -> Option<Self::Output> {
                let shape = self.shape()?;
                let ($($a,)*) = self.0;
                Some(($(
                    $a.build_with_shape(shape)?,
                )*))
            }
        }

        impl<$($A,)* S: Shape> BuildWithShape<S> for Broadcast<($($A,)*)>
        where
            Self: Shaped,
            $($A: BuildWithShape<S>,)*
        {
            type Output = ($(<$A as BuildWithShape<S>>::Output,)*);

            fn build_with_shape(self, shape: S) -> Option<Self::Output> {
                let ($($a,)*) = self.0;
                Some(($(
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

impl<'a, S: Shape, D: ?Sized> View<'a, S, D> {
    pub fn broadcast_into_fail<S2: Shape>(self, shape: S2) -> View<'a, S2, D>
    where
        S: IntoIndex<S2>,
    {
        self.shape.broadcast_into_fail(shape);
        View {
            shape,
            offset: self.offset,
            strides: S::into_index(self.strides),
            data: self.data,
        }
    }
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

impl<'a, S: Shape, D: ?Sized> ViewMut<'a, S, D> {
    pub fn broadcast_into_fail<S2: Shape>(self, shape: S2) -> ViewMut<'a, S2, D>
    where
        S: IntoIndex<S2> + BroadcastIntoNoAlias<S2>,
    {
        self.shape.broadcast_into_fail(shape);
        ViewMut {
            shape,
            offset: self.offset,
            strides: S::into_index(self.strides),
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
        S2: IntoIndex<S> + BroadcastIntoNoAlias<S>,
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
            strides: S2::into_index(self.array.strides),
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

#[cfg(test)]
mod test {

    use crate::{
        alloc, alloc_shape, Array, AsIndex, Broadcast, BroadcastTogether, Build, Const,
        DefiniteRange, NewAxis, OutMarker, OutTarget, Shape, View,
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

        fn bernstein_coef<A, B, O: OutTarget<Shape = S, Data = [f32]>, S: Shape>(
            c_m: &A,
            out: B,
        ) -> O::Output
        where
            for<'a> Broadcast<(&'a A, OutMarker<B>)>: Build<Output = (View<'a, S, [f32]>, O)>,
        {
            let (c_m, mut out_target) = Broadcast((c_m, OutMarker(out)))
                .build()
                .expect("Could not broadcast arrays together");
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

            out_target.output()
        }

        // TEST DATA

        let a = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
            offset: 0,
            data: vec![1., 2., 3., 4.],
        };

        let mut b = Array {
            shape: (2, 2),
            strides: (2, 2).default_strides(),
            offset: 0,
            data: vec![0.; 4],
        };

        bernstein_coef(&a, &mut b);

        dbg!(bernstein_coef(&a, alloc()));
        dbg!(bernstein_coef(&a, alloc_shape((2, 2))));

        bernstein_coef(&a, &mut b);
    }

    #[test]
    fn test_sum_prod() {
        fn sum_prod<A, B, S: Shape>(in1: &A, in2: &B) -> f32
        where
            for<'a> Broadcast<(&'a A, &'a B)>:
                Build<Output = (View<'a, S, [f32]>, View<'a, S, [f32]>)>,
        {
            let (in1, in2) = Broadcast((in1, in2))
                .build()
                .expect("Arrays cannot be broadcast together");

            in1.into_iter()
                .zip(in2.into_iter())
                .map(|(a, b)| a * b)
                .sum()
        }

        let a = Array {
            shape: (Const::<2>, Const::<2>),
            strides: (Const::<2>, Const::<2>).default_strides(),
            offset: 0,
            data: vec![1., 2., 3., 4.],
        };

        sum_prod(&a, &a);
    }

    #[test]
    fn test_add() {
        fn add<A, B, C, O: OutTarget<Shape = S, Data = [f32]>, S: Shape>(
            a: &A,
            b: &B,
            out: C,
        ) -> O::Output
        where
            for<'a> Broadcast<(&'a A, &'a B, OutMarker<C>)>:
                Build<Output = (View<'a, S, [f32]>, View<'a, S, [f32]>, O)>,
        {
            let (a, b, mut out_target) = Broadcast((a, b, OutMarker(out)))
                .build()
                .expect("Arrays cannot be broadcast together");
            let mut out = out_target.view_mut();

            for (out, (a, b)) in (&mut out).into_iter().zip(a.into_iter().zip(b.into_iter())) {
                *out = a + b;
            }

            out_target.output()
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

        add(&a, &b, &mut c);

        assert_eq!(c.data, [11., 12., 21., 22., 11., 12., 21., 22.]);
    }

    #[test]
    fn test_sum() {
        fn sum<A, S: Shape>(a: &A) -> f32
        where
            for<'a> &'a A: Build<Output = View<'a, S, [f32]>>,
        {
            let a = a.build().unwrap(); // Infalliable?

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
            OutMarker<A>: Build<Output = O>,
        {
            let mut out_target = OutMarker(out).build().unwrap(); // Infalliable?
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
        assert_eq!(
            ((Const::<3>, Const::<4>), (Const::<3>, Const::<4>))
                .broadcast_together()
                .unwrap()
                .as_index(),
            [3, 4]
        );
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
            .is_none());
        assert!(((Const::<3>, Const::<4>), (3, 5))
            .broadcast_together()
            .is_none());

        assert_eq!(
            ((Const::<3>, Const::<4>), ())
                .broadcast_together()
                .unwrap()
                .as_index(),
            [3, 4]
        );
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
