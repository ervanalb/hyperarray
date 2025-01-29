use std::fmt;
use std::marker;
use std::ops;

/// In a [Shape] tuple, this represents an axis length known at compile time.
/// The primitive type `usize` is used for a dimension not known at compile time,
/// or [NewAxis] for a unit-length broadcastable axis.
/// (Unlike with NumPy, an axis length `Const::<1>` or `1_usize` will not broadcast)
#[derive(Default, Clone, Copy)]
pub struct Const<const N: usize>;

impl<const N: usize> fmt::Debug for Const<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", N)
    }
}

/// A tuple where each element represents the length of an axis of multi-dimensional data.
/// Each entry is [Const<N>](Const).
///
/// // TODO write a shape example
pub trait Shape {
    type Index: Index;
    const SHAPE: Self::Index;
    const STRIDES: Self::Index;
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

    /// Panics if the given index is out-of-bounds for data of this shape.
    ///
    /// ```
    /// use nada::Index;
    ///
    /// [1, 5].out_of_bounds_fail(&[2, 6]); // No panic, since 1 < 2 and 5 < 6
    /// ```
    fn out_of_bounds_fail(&self, shape: &Self) {
        for (i, s) in self.into_iter().zip(shape.into_iter()) {
            if i >= s {
                panic!("Index out of bounds: index={self:?} shape={shape:?}");
            }
        }
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
// Shape

impl Shape for () {
    type Index = [usize; 0];
    const SHAPE: Self::Index = [];
    const STRIDES: Self::Index = [];
}

impl<const N0: usize> Shape for (Const<N0>,) {
    type Index = [usize; 1];
    const SHAPE: Self::Index = [N0];
    const STRIDES: Self::Index = [1];
}

impl<const N0: usize, const N1: usize> Shape for (Const<N0>, Const<N1>) {
    type Index = [usize; 2];
    const SHAPE: Self::Index = [N0, N1];
    const STRIDES: Self::Index = [N1, 1];
}

// TODO turn the above into macro

/*
macro_rules! impl_shape {
    ($($a:ident $A:ident)*) => {
        impl<$($A,)* const N: usize,> Shape for ($($A,)* Const<N>,)
            where ($($A,)*): Shape
        {
            type Index = <<($($A,)*) as Shape>::Index as Push>::OneBigger;

            fn AS_INDEX -> Self::Index {
                <($($A,)*) as Shape>::AS_INDEX.append(N)
            }
        }
    };
}

impl_shape!();
impl_shape!(a0 A0);
impl_shape!(a0 A0 a1 A1);
*/

// Into

impl<'a, S: Shape, D: ?Sized + 'a, D2: ops::Deref<Target = D>> From<&'a Array<S, D2>>
    for View<'a, S, D>
{
    fn from(value: &'a Array<S, D2>) -> Self {
        View {
            _shape: marker::PhantomData,
            data: &value.data,
        }
    }
}

impl<'a, S: Shape, D: ?Sized> From<&'a View<'a, S, D>> for View<'a, S, D> {
    fn from(value: &'a View<'a, S, D>) -> Self {
        View {
            _shape: marker::PhantomData,
            data: value.data,
        }
    }
}

impl<'a, S: Shape, D: ?Sized> From<&'a ViewMut<'a, S, D>> for View<'a, S, D> {
    fn from(value: &'a ViewMut<'a, S, D>) -> Self {
        View {
            _shape: marker::PhantomData,
            data: value.data,
        }
    }
}

impl<'a, S: Shape, D: ?Sized + 'a, D2: ops::Deref<Target = D> + ops::DerefMut<Target = D>>
    From<&'a mut Array<S, D2>> for ViewMut<'a, S, D>
{
    fn from(value: &'a mut Array<S, D2>) -> Self {
        ViewMut {
            _shape: marker::PhantomData,
            data: &mut value.data,
        }
    }
}

impl<'a, S: Shape, D: ?Sized> From<&'a mut ViewMut<'a, S, D>> for ViewMut<'a, S, D> {
    fn from(value: &'a mut ViewMut<'a, S, D>) -> Self {
        ViewMut {
            _shape: marker::PhantomData,
            data: value.data,
        }
    }
}

//////////////////////////////////////////////////////

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

impl<I: Index> DefiniteRange for ops::RangeTo<I> {
    type Item = I;

    fn first(&self) -> Option<Self::Item> {
        self.end.into_iter().all(|n| n > 0).then_some(I::zero())
    }

    fn next(&self, mut cur: Self::Item) -> Option<Self::Item> {
        for (i, n) in cur.iter_mut().rev().zip(self.end.into_iter().rev()) {
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

impl<I: Index> DefiniteRange for ops::RangeToInclusive<I> {
    type Item = I;

    fn first(&self) -> Option<Self::Item> {
        Some(I::zero())
    }

    fn next(&self, mut cur: Self::Item) -> Option<Self::Item> {
        for (i, n) in cur.iter_mut().rev().zip(self.end.into_iter().rev()) {
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

impl<I: Index> DefiniteRange for ops::Range<I> {
    type Item = I;

    fn first(&self) -> Option<Self::Item> {
        self.start
            .into_iter()
            .zip(self.end.into_iter())
            .all(|(s, e)| e > s)
            .then_some(self.start)
    }

    fn next(&self, mut cur: Self::Item) -> Option<Self::Item> {
        for (i, (s, e)) in cur
            .iter_mut()
            .rev()
            .zip(self.start.into_iter().rev().zip(self.end.into_iter().rev()))
        {
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

impl<I: Index> DefiniteRange for ops::RangeInclusive<I> {
    type Item = I;

    fn first(&self) -> Option<Self::Item> {
        self.start()
            .into_iter()
            .zip(self.end().into_iter())
            .all(|(s, e)| e >= s)
            .then_some(*self.start())
    }

    fn next(&self, mut cur: Self::Item) -> Option<Self::Item> {
        for (i, (s, e)) in cur.iter_mut().rev().zip(
            self.start()
                .into_iter()
                .rev()
                .zip(self.end().into_iter().rev()),
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
    shape: marker::PhantomData<S>,
    data: D,
}

/// A view into multi-dimensional data.
///
/// This type holds an immutable reference the underlying data.
/// For a mutable reference, see [ViewMut],
/// or for owned data, see [Array].
#[derive(Debug)]
pub struct View<'a, S: Shape, D: ?Sized> {
    _shape: marker::PhantomData<S>,
    data: &'a D,
}

/// A mutable view into multi-dimensional data.
///
/// This type holds an mutable reference the underlying data.
/// For an immutable reference, see [View],
/// or for owned data, see [Array].
#[derive(Debug)]
pub struct ViewMut<'a, S: Shape, D: ?Sized> {
    _shape: marker::PhantomData<S>,
    data: &'a mut D,
}

/////////////////////////////////////////////

impl<S: Shape, E> View<'_, S, [E]> {
    pub unsafe fn get_unchecked(&self, idx: S::Index) -> &E {
        self.data.get_unchecked(idx.to_i(&S::STRIDES))
    }
}

impl<S: Shape, E> ops::Index<S::Index> for View<'_, S, [E]> {
    type Output = E;
    fn index(&self, idx: S::Index) -> &E {
        idx.out_of_bounds_fail(&S::SHAPE);
        unsafe { self.get_unchecked(idx) }
    }
}
impl<'a, S: Shape, E> IntoIterator for &'a View<'_, S, [E]> {
    type Item = &'a E;
    type IntoIter = NdIter<'a, S, E>;
    fn into_iter(self) -> Self::IntoIter {
        NdIter {
            shape: marker::PhantomData,
            data: &self.data[..],
            idx: (..S::SHAPE).first(),
        }
    }
}
impl<S: Shape, E> ViewMut<'_, S, [E]> {
    pub unsafe fn get_unchecked(&self, idx: S::Index) -> &E {
        self.data.get_unchecked(idx.to_i(&S::STRIDES))
    }
}
impl<S: Shape, E> ops::Index<S::Index> for ViewMut<'_, S, [E]> {
    type Output = E;
    fn index(&self, idx: S::Index) -> &E {
        idx.out_of_bounds_fail(&S::SHAPE);
        unsafe { self.get_unchecked(idx) }
    }
}
impl<'a, S: Shape, E> IntoIterator for &'a ViewMut<'_, S, [E]> {
    type Item = &'a E;
    type IntoIter = NdIter<'a, S, E>;
    fn into_iter(self) -> Self::IntoIter {
        NdIter {
            shape: marker::PhantomData,
            data: &self.data[..],
            idx: (..S::SHAPE).first(),
        }
    }
}
impl<S: Shape, E, D: ops::Deref<Target = [E]>> Array<S, D> {
    pub unsafe fn get_unchecked(&self, idx: S::Index) -> &E {
        self.data.as_ref().get_unchecked(idx.to_i(&S::STRIDES))
    }
}
impl<S: Shape, E, D: ops::Deref<Target = [E]>> ops::Index<S::Index> for Array<S, D> {
    type Output = E;
    fn index(&self, idx: S::Index) -> &E {
        idx.out_of_bounds_fail(&S::SHAPE);
        unsafe { self.get_unchecked(idx) }
    }
}
impl<'a, S: Shape, E: 'a, D: ops::Deref<Target = [E]>> IntoIterator for &'a Array<S, D> {
    type Item = &'a E;
    type IntoIter = NdIter<'a, S, E>;
    fn into_iter(self) -> Self::IntoIter {
        NdIter {
            shape: marker::PhantomData,
            data: &self.data[..],
            idx: (..S::SHAPE).first(),
        }
    }
}

impl<S: Shape, E> ViewMut<'_, S, [E]> {
    pub unsafe fn get_unchecked_mut(&mut self, idx: S::Index) -> &mut E {
        self.data.get_unchecked_mut(idx.to_i(&S::STRIDES))
    }
}
impl<S: Shape, E> ops::IndexMut<S::Index> for ViewMut<'_, S, [E]> {
    fn index_mut(&mut self, idx: S::Index) -> &mut E {
        idx.out_of_bounds_fail(&S::SHAPE);
        unsafe { self.get_unchecked_mut(idx) }
    }
}
impl<'a, S: Shape, E> IntoIterator for &'a mut ViewMut<'_, S, [E]> {
    type Item = &'a mut E;
    type IntoIter = NdIterMut<'a, S, E>;
    fn into_iter(self) -> Self::IntoIter {
        NdIterMut {
            shape: marker::PhantomData,
            data: &mut self.data[..],
            idx: (..S::SHAPE).first(),
        }
    }
}
impl<S: Shape, E, D: ops::Deref<Target = [E]> + ops::DerefMut<Target = [E]>> Array<S, D> {
    pub unsafe fn get_unchecked_mut(&mut self, idx: S::Index) -> &mut E {
        self.data.as_mut().get_unchecked_mut(idx.to_i(&S::STRIDES))
    }
}
impl<S: Shape, E, D: ops::Deref<Target = [E]> + ops::DerefMut<Target = [E]>> ops::IndexMut<S::Index>
    for Array<S, D>
{
    fn index_mut(&mut self, idx: S::Index) -> &mut E {
        idx.out_of_bounds_fail(&S::SHAPE);
        unsafe { self.get_unchecked_mut(idx) }
    }
}
impl<'a, S: Shape, E: 'a, D: ops::DerefMut<Target = [E]>> IntoIterator for &'a mut Array<S, D> {
    type Item = &'a mut E;
    type IntoIter = NdIterMut<'a, S, E>;
    fn into_iter(self) -> Self::IntoIter {
        NdIterMut {
            shape: marker::PhantomData,
            data: &mut self.data[..],
            idx: (..S::SHAPE).first(),
        }
    }
}

/////////////////////////////////////////////

pub struct NdIter<'a, S: Shape, E> {
    shape: marker::PhantomData<S>,
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

        let val = unsafe { self.data.get_unchecked(idx.to_i(&S::STRIDES)) };
        self.idx = (..S::SHAPE).next(idx);
        Some(val)
    }
}

pub struct NdIterMut<'a, S: Shape, E> {
    shape: marker::PhantomData<S>,
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
        let val = unsafe { &mut *(self.data.get_unchecked_mut(idx.to_i(&S::STRIDES)) as *mut E) };
        self.idx = (..S::SHAPE).next(idx);
        Some(val)
    }
}

pub struct NdEnumerate<I>(I);

impl<'a, S: Shape, E> Iterator for NdEnumerate<NdIter<'a, S, E>> {
    type Item = (S::Index, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.0.idx?;
        let val = unsafe { self.0.data.get_unchecked(idx.to_i(&S::STRIDES)) };
        self.0.idx = (..S::SHAPE).next(idx);
        Some((idx, val))
    }
}

impl<'a, S: Shape, E> Iterator for NdEnumerate<NdIterMut<'a, S, E>> {
    type Item = (S::Index, &'a mut E);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.0.idx?;
        let val = unsafe { &mut *(self.0.data.get_unchecked_mut(idx.to_i(&S::STRIDES)) as *mut E) };
        self.0.idx = (..S::SHAPE).next(idx);
        Some((idx, val))
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

    use crate::{Array, Const, DefiniteRange, Shape, View, ViewMut};
    use std::marker;

    #[test]
    fn test_iter() {
        let mut t = Array {
            shape: marker::PhantomData::<(Const<2>, Const<2>)>,
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

        fn bernstein_coef<A, O, S: Shape>(c_m: &A, out: &mut O)
        where
            for<'a> &'a A: Into<View<'a, S, [f32]>>,
            for<'a> &'a mut O: Into<ViewMut<'a, S, [f32]>>,
        {
            let c_m = c_m.into();
            let mut out = out.into();

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
                        let den: usize = S::SHAPE
                            .into_iter()
                            .zip(j.into_iter())
                            .map(|(d_n, j_n)| binomial(d_n, j_n))
                            .product();
                        let b = (num as f32) / (den as f32);
                        b * c_m[j]
                    })
                    .sum();
            }
        }

        // TEST DATA

        let a = Array {
            shape: marker::PhantomData::<(Const<2>, Const<2>)>,
            data: vec![1., 2., 3., 4.],
        };

        let mut b = Array {
            shape: marker::PhantomData::<(Const<2>, Const<2>)>,
            data: vec![0.; 4],
        };

        bernstein_coef(&a, &mut b);
    }

    #[test]
    fn test_sum_prod() {
        fn sum_prod<A, B, S: Shape>(in1: &A, in2: &B) -> f32
        where
            for<'a> &'a A: Into<View<'a, S, [f32]>>,
            for<'a> &'a B: Into<View<'a, S, [f32]>>,
        {
            let in1 = in1.into();
            let in2 = in2.into();

            in1.into_iter()
                .zip(in2.into_iter())
                .map(|(a, b)| a * b)
                .sum()
        }

        let a = Array {
            shape: marker::PhantomData::<(Const<2>, Const<2>)>,
            data: vec![1., 2., 3., 4.],
        };

        sum_prod(&a, &a);
    }

    #[test]
    fn test_add() {
        fn add<A, B, O, S: Shape>(a: &A, b: &B, out: &mut O)
        where
            for<'a> &'a A: Into<View<'a, S, [f32]>>,
            for<'a> &'a B: Into<View<'a, S, [f32]>>,
            for<'a> &'a mut O: Into<ViewMut<'a, S, [f32]>>,
        {
            let a = a.into();
            let b = b.into();
            let mut out = out.into();

            for (out, (a, b)) in (&mut out).into_iter().zip(a.into_iter().zip(b.into_iter())) {
                *out = a + b;
            }
        }

        let a = Array {
            shape: marker::PhantomData::<(Const<2>, Const<2>)>,
            data: vec![1., 2., 3., 4.],
        };

        let b = Array {
            shape: marker::PhantomData::<(Const<2>, Const<2>)>,
            data: vec![10., 20., 30., 40.],
        };

        let mut c = Array {
            shape: marker::PhantomData::<(Const<2>, Const<2>)>,
            data: vec![0.; 4],
        };

        add(&a, &b, &mut c);

        assert_eq!(c.data, [11., 22., 33., 44.]);
    }

    #[test]
    fn test_sum() {
        fn sum<A, S: Shape>(a: &A) -> f32
        where
            for<'a> &'a A: Into<View<'a, S, [f32]>>,
        {
            let a = a.into();

            a.into_iter().sum()
        }

        let a = Array {
            shape: marker::PhantomData::<(Const<2>, Const<2>)>,
            data: vec![1., 2., 3., 4.],
        };

        sum(&a);
    }

    #[test]
    fn test_ones() {
        fn ones<O, S: Shape>(out: &mut O)
        where
            for<'a> &'a mut O: Into<ViewMut<'a, S, [f32]>>,
        {
            let mut out = out.into();

            for e in &mut out {
                *e = 1.;
            }
        }

        let mut a = Array {
            shape: marker::PhantomData::<(Const<2>, Const<2>)>,
            data: vec![1., 2., 3., 4.],
        };

        ones(&mut a);
    }
}
