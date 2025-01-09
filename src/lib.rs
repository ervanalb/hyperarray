use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// Represents a single dimension of a multi dimensional shape
pub trait Dim: 'static + Sized + Clone + Copy + fmt::Debug {
    fn value(&self) -> usize;
}

/// Represents a dimension whose length is known at compile time
#[derive(Default, Clone, Copy)]
pub struct Const<const N: usize>;

impl<const N: usize> fmt::Debug for Const<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Const<{}>", N)
    }
}

impl<const M: usize> Dim for Const<M> {
    fn value(&self) -> usize {
        M
    }
}

impl Dim for usize {
    fn value(&self) -> usize {
        *self
    }
}

/// The shape of an ndarray, e.g. a 2D statically-sized array might have shape (Const<3>, Const<2>)
/// or a 3D dynamically-sized array might have shape (3, 4, 5)
pub trait Shape: 'static + Sized + Clone + Copy + fmt::Debug {
    /// This type is [usize; Self::NUM_DIMS] but rust doesn't allow that :(
    type Index: NdIndex;

    // Required methods:

    fn value(&self) -> Self::Index;

    // Provided methods:

    fn num_elements(&self) -> usize {
        self.value().into_iter().product()
    }

    fn default_strides(&self) -> Self::Index {
        let mut result = Self::Index::zero();
        let mut acc = 1;
        for (i, d) in self.value().into_iter().enumerate().rev() {
            result[i] = acc;
            acc *= d;
        }
        result
    }
}

pub trait NdIndex:
    Clone
    + Copy
    + fmt::Debug
    + IntoIterator<Item = usize, IntoIter: DoubleEndedIterator + ExactSizeIterator>
    + Index<usize, Output = usize>
    + IndexMut<usize>
{
    // Required methods:
    fn zero() -> Self;
    fn iter_mut<'a>(&'a mut self) -> std::slice::IterMut<'a, usize>;

    // Provided methods:
    fn in_bounds<S: Shape<Index = Self>>(&self, shape: &S) -> bool {
        self.into_iter()
            .zip(shape.value().into_iter())
            .all(|(i, s)| i < s)
    }

    fn out_of_bounds_fail<S: Shape<Index = Self>>(&self, shape: &S) {
        for (i, s) in self.into_iter().zip(shape.value().into_iter()) {
            if i >= s {
                panic!("Index out of bounds: index={self:?} shape={shape:?}");
            }
        }
    }

    fn to_i(&self, strides: &Self) -> usize {
        strides
            .into_iter()
            .zip(self.into_iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    fn next_index<S: Shape<Index = Self>>(mut self, shape: &S) -> Option<S::Index> {
        for (i, n) in self.iter_mut().rev().zip(shape.value().into_iter().rev()) {
            *i += 1;
            match (*i).cmp(&n) {
                Ordering::Less => {
                    // In-bounds
                    return Some(self);
                }
                Ordering::Equal => {
                    // Iteration finished in this axis
                    *i = 0;
                }
                Ordering::Greater => {
                    // Out-of-bounds
                    panic!("Index out of bounds: index={self:?} shape={shape:?}");
                }
            }
        }
        None
    }
}

impl<const N: usize> NdIndex for [usize; N] {
    fn zero() -> Self {
        [0; N]
    }

    fn iter_mut<'a>(&'a mut self) -> std::slice::IterMut<'a, usize> {
        <[usize]>::iter_mut(self)
    }
}

impl<D1: Dim> Shape for (D1,) {
    type Index = [usize; 1];

    fn value(&self) -> Self::Index {
        [self.0.value()]
    }
}

impl<D1: Dim, D2: Dim> Shape for (D1, D2) {
    type Index = [usize; 2];

    fn value(&self) -> Self::Index {
        [self.0.value(), self.1.value()]
    }
}

/////////////////////////////////////////////

#[derive(Debug)]
pub struct NdViewParameters<S: Shape, E> {
    shape: S,
    element: PhantomData<E>,
    offset: usize,
    strides: S::Index,
}

impl<S: Shape, E> Clone for NdViewParameters<S, E> {
    fn clone(&self) -> NdViewParameters<S, E> {
        NdViewParameters {
            shape: self.shape,
            element: PhantomData,
            offset: self.offset,
            strides: self.strides,
        }
    }
}

impl<S: Shape, E> NdViewParameters<S, E> {
    pub fn from_array<D>(t: &NdArray<S, E, D>) -> NdViewParameters<S, E> {
        NdViewParameters {
            shape: t.shape,
            element: PhantomData,
            offset: 0,
            strides: S::default_strides(&t.shape),
        }
    }
}

// Wrappers for parameters + data
// Three kinds: owned, ref, mut

#[derive(Debug, Clone)]
pub struct NdArray<S: Shape, E, D> {
    shape: S,
    element: PhantomData<E>,
    data: D,
}

#[derive(Debug)]
pub struct NdView<'a, S: Shape, E, D> {
    parameters: NdViewParameters<S, E>,
    data: &'a D,
}

#[derive(Debug)]
pub struct NdViewMut<'a, S: Shape, E, D> {
    parameters: NdViewParameters<S, E>,
    data: &'a mut D,
}

// Get a view
pub trait View {
    type Shape: Shape;
    type Element;
    type Data;

    fn parameters(&self) -> NdViewParameters<Self::Shape, Self::Element>;
    fn data(&self) -> &Self::Data;

    fn view(&self) -> NdView<Self::Shape, Self::Element, Self::Data> {
        NdView {
            parameters: self.parameters(),
            data: self.data(),
        }
    }
}

impl<S: Shape, E, D> View for NdArray<S, E, D> {
    type Shape = S;
    type Element = E;
    type Data = D;

    fn parameters(&self) -> NdViewParameters<Self::Shape, Self::Element> {
        NdViewParameters::from_array(self)
    }

    fn data(&self) -> &D {
        &self.data
    }
}

impl<S: Shape, E, D> View for NdView<'_, S, E, D> {
    type Shape = S;
    type Element = E;
    type Data = D;

    fn parameters(&self) -> NdViewParameters<Self::Shape, Self::Element> {
        self.parameters.clone()
    }

    fn data(&self) -> &D {
        self.data
    }
}

impl<S: Shape, E, D> View for NdViewMut<'_, S, E, D> {
    type Shape = S;
    type Element = E;
    type Data = D;

    fn parameters(&self) -> NdViewParameters<Self::Shape, Self::Element> {
        self.parameters.clone()
    }

    fn data(&self) -> &D {
        self.data
    }
}

// Get a mutable view
pub trait ViewMut: View {
    fn data_mut(&mut self) -> &mut Self::Data;

    fn view_mut(&mut self) -> NdViewMut<'_, Self::Shape, Self::Element, Self::Data> {
        NdViewMut {
            parameters: self.parameters(),
            data: self.data_mut(),
        }
    }
}

impl<S: Shape, E, D> ViewMut for NdArray<S, E, D> {
    fn data_mut(&mut self) -> &mut D {
        &mut self.data
    }
}

impl<S: Shape, E, D> ViewMut for NdViewMut<'_, S, E, D> {
    fn data_mut(&mut self) -> &mut D {
        self.data
    }
}

/////////////////////////////////////////////

impl<S: Shape, E> NdViewParameters<S, E> {
    unsafe fn get_unchecked<'a>(&self, data: &'a [E], index: S::Index) -> &'a E {
        data.get_unchecked(self.offset + index.to_i(&self.strides))
    }

    unsafe fn get_unchecked_mut<'a>(&mut self, data: &'a mut [E], index: S::Index) -> &'a mut E {
        data.get_unchecked_mut(self.offset + index.to_i(&self.strides))
    }

    fn index<'a>(&self, data: &'a [E], index: S::Index) -> &'a E {
        index.out_of_bounds_fail(&self.shape);
        unsafe { self.get_unchecked(data, index) }
    }

    fn index_mut<'a>(&mut self, data: &'a mut [E], index: S::Index) -> &'a mut E {
        index.out_of_bounds_fail(&self.shape);
        unsafe { self.get_unchecked_mut(data, index) }
    }
}

macro_rules! impl_view_methods {
    ($struct:ident $(<$lt:lifetime>)?) => {
        impl<S: Shape, E, D: AsRef<[E]>> $struct<$($lt,)? S, E, D> {
            pub unsafe fn get_unchecked(&self, idx: S::Index) -> &E {
                self.parameters().get_unchecked(self.data().as_ref(), idx)
            }
        }

        impl<S: Shape, E, D: AsRef<[E]>> Index<S::Index> for $struct<$($lt,)? S, E, D> {
            type Output = E;

            fn index(&self, idx: S::Index) -> &E {
                self.parameters().index(self.data().as_ref(), idx)
            }
        }

        impl<'a, S: Shape, E, D: AsRef<[E]>> IntoIterator for &'a $struct<$($lt,)? S, E, D> {
            type Item = &'a E;
            type IntoIter = NdIter<'a, S, E>;

            fn into_iter(self) -> Self::IntoIter {
                NdIter::new(self.parameters(), self.data().as_ref())
            }
        }
    };
}

macro_rules! impl_view_mut_methods {
    ($struct:ident $(<$lt:lifetime>)?) => {
        impl<S: Shape, E, D: AsRef<[E]> + AsMut<[E]>> $struct<$($lt,)? S, E, D> {
            pub unsafe fn get_unchecked_mut(&mut self, idx: S::Index) -> &E {
                self.parameters().get_unchecked_mut(self.data_mut().as_mut(), idx)
            }
        }

        impl<S: Shape, E, D: AsRef<[E]> + AsMut<[E]>> IndexMut<S::Index> for $struct<$($lt,)? S, E, D> {
            fn index_mut(&mut self, idx: S::Index) -> &mut E {
                self.parameters().index_mut(self.data_mut().as_mut(), idx)
            }
        }

        impl<'a, S: Shape, E, D: AsRef<[E]> + AsMut<[E]>> IntoIterator for &'a mut $struct<$($lt,)? S, E, D> {
            type Item = &'a mut E;
            type IntoIter = NdIterMut<'a, S, E>;

            fn into_iter(self) -> Self::IntoIter {
                NdIterMut::new(self.parameters(), self.data_mut().as_mut())
            }
        }
    };
}

impl_view_methods!(NdView<'_>);
impl_view_methods!(NdViewMut<'_>);
impl_view_methods!(NdArray);

impl_view_mut_methods!(NdViewMut<'_>);
impl_view_mut_methods!(NdArray);

/////////////////////////////////////////////

pub struct NdIter<'a, S: Shape, E> {
    parameters: NdViewParameters<S, E>,
    data: &'a [E],
    idx: Option<S::Index>,
}

impl<'a, S: Shape, E> NdIter<'a, S, E> {
    fn new(parameters: NdViewParameters<S, E>, data: &'a [E]) -> Self {
        let idx = (parameters.shape.num_elements() > 0).then_some(S::Index::zero());
        NdIter {
            parameters,
            data,
            idx,
        }
    }
}

impl<'a, S: Shape, E> Iterator for NdIter<'a, S, E> {
    type Item = &'a E;

    fn next(&mut self) -> Option<Self::Item> {
        self.idx.map(|idx| {
            let val = unsafe { self.parameters.get_unchecked(self.data, idx) };
            self.idx = idx.next_index(&self.parameters.shape);
            val
        })
    }
}

pub struct NdIterMut<'a, S: Shape, E> {
    parameters: NdViewParameters<S, E>,
    data: &'a mut [E],
    idx: Option<S::Index>,
}

impl<'a, S: Shape, E> NdIterMut<'a, S, E> {
    fn new(parameters: NdViewParameters<S, E>, data: &'a mut [E]) -> Self {
        let idx = (parameters.shape.num_elements() > 0).then_some(S::Index::zero());
        NdIterMut {
            parameters,
            data,
            idx,
        }
    }
}

impl<'a, S: Shape, E> Iterator for NdIterMut<'a, S, E> {
    type Item = &'a mut E;

    fn next(&mut self) -> Option<Self::Item> {
        self.idx.map(|idx| {
            let val =
                unsafe { &mut *(self.parameters.get_unchecked_mut(self.data, idx) as *mut E) };
            self.idx = idx.next_index(&self.parameters.shape);
            val
        })
    }
}

/*
impl<'a, S: Shape, E, D: AsRef<[E]>> IntoIterator for &'a TensorView<'_, S, E, D> {
    type Item = &'a E;
    type IntoIter = TensorIter<'a, S, E, D>;

    fn into_iter(self) -> Self::IntoIter {
        TensorIter::new(self.parameters(), self.data())
    }
}

impl<'a, S: Shape, E, D: AsRef<[E]>> IntoIterator for &'a Tensor<S, E, D> {
    type Item = &'a E;
    type IntoIter = TensorIter<'a, S, E, D>;

    fn into_iter(self) -> Self::IntoIter {
        TensorIter::new(self.parameters(), self.data())
    }
}
*/

/*
impl<'a, S: Shape, E, D: AsRef<[E]>> IntoIterator for &'a TensorViewMut<'a, S, E, D> {
    type Item = &'a E;
    type IntoIter = TensorIter<'a, S, E, D>;

    fn into_iter(self) -> Self::IntoIter {
        TensorView {
            parameters: self.parameters.clone(),
            data: self.data,
        }
        .iter()
    }
}
*/

pub fn bernstein_coef<S: Shape, E, DIN, DOUT>(c_m: NdArray<S, E, DIN>) -> NdArray<S, E, DOUT> {
    todo!();
    /*
    (tensor_product(D::zero()..=i))
        .map(|j| {
            let num: usize = i
                .into_iter()
                .zip(j.into_iter())
                .map(|(i_n, j_n)| binomial(i_n, j_n))
                .product();
            let den: usize = D::shape()
                .into_iter()
                .zip(j.into_iter())
                .map(|(d_n, j_n)| binomial(d_n, j_n))
                .product();
            (num as f32) / (den as f32)
        })
        .sum()
    */
}

#[cfg(test)]
mod test {

    use crate::{Const, NdArray};
    use std::marker::PhantomData;

    #[test]
    fn test() {
        let mut t = NdArray {
            shape: (Const::<2>, Const::<2>),
            element: PhantomData::<i32>,
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
}

fn main() {
    /*
    struct Vector<const N: usize>([f32; N]);
    impl<const N: usize> Tensor for Vector<N> {
        type Shape = (Const<N>,);
        type Item = f32;

        fn shape(&self) -> Self::Shape {
            Default::default()
        }
    }
    impl<const N: usize> Index<[usize; 1]> for Vector<N> {
        type Output = f32;
        fn index(&self, i: [usize; 1]) -> &Self::Output {
            &self.0[i[0]]
        }
    }

    impl<const N: usize> Neg for Vector<N> {
        type Output = Vector<N>;

        fn neg(self) -> Self::Output {
            for item in self.iter_mut() {
                item = -item;
            }
            self
        }
    }

    impl<const N: usize> Add<Vector<N>> for Vector<N> {
        type Output = Vector<N>;
        fn add(self, other: Vector<N>) {
            Vector<N>::from_fn(|i| )
        }
    }

    struct Matrix<const N: usize, const M: usize>([[f32; M]; N]);
    impl<const N: usize, const M: usize> Tensor for Matrix<N, M> {
        type Shape = (Const<N>, Const<M>);
        type Item = f32;
        fn shape(&self) -> Self::Shape {
            Default::default()
        }
    }
    impl<const N: usize, const M: usize> Index<[usize; 2]> for Matrix<N, M> {
        type Output = f32;
        fn index(&self, i: [usize; 2]) -> &Self::Output {
            &self.0[i[0]][i[1]]
        }
    }
    */

    let mut t = NdArray {
        shape: (Const::<2>, Const::<2>),
        element: PhantomData::<i32>,
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

    /*
    let mut t2 = t.view_mut();

    let t3 = &mut t2;

    t3[[1, 0]] = 5;

    //dbg!(&t2);
    //dbg!(t2[[1, 0]]);
    //dbg!(t2[[1, 0]]);
    //t2[[1, 0]] = 5;
    //dbg!(t2[[1, 0]]);
    //

    let mut i = [0; 2];
    let shape = (Const::<3>, Const::<4>);
    for _ in 0..20 {
        dbg!(i);
        i = i.next_index(&shape).unwrap();
    }
    */
}
