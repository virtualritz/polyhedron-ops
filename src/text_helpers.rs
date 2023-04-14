use num_traits::{Float, FromPrimitive};
use std::fmt::{Display, Write};

#[inline]
pub(crate) fn format_float<T>(x: T) -> String
where
    T: Float + FromPrimitive + Display,
{
    let int_part = x.trunc();
    let decimal_part = x - int_part;

    if decimal_part.abs() < T::epsilon() {
        format!("{}", int_part.to_i64().unwrap())
    } else {
        let decimal_digits = (-x.abs().log10().floor()).to_usize().unwrap();
        format!("{:.*}", decimal_digits, x)
    }
}

#[test]
fn test_format_float() {
    assert_eq!(format_float(1.0), "1");
    assert_eq!(format_float(-0.2), "-0.2");
    assert_eq!(format_float(0.30), "0.3");
}

pub(crate) fn _format_float_slice<T>(slice: &[T]) -> String
where
    T: Float + FromPrimitive + Display,
{
    if slice.is_empty() {
        String::new()
    } else {
        let mut string = String::with_capacity(slice.len() * 2);
        if 1 == slice.len() {
            write!(&mut string, "{}", format_float(slice[0])).unwrap();
        } else {
            string.push('[');
            write!(&mut string, "{}", format_float(slice[0])).unwrap();
            for i in slice.get(1..).unwrap() {
                write!(&mut string, ",{}", format_float(*i)).unwrap();
            }
            string.push(']');
        }
        string
    }
}

pub(crate) fn format_integer_slice<T>(slice: &[T]) -> String
where
    T: Display,
{
    if slice.is_empty() {
        String::new()
    } else {
        let mut string = String::with_capacity(slice.len() * 2);
        if 1 == slice.len() {
            write!(&mut string, "{}", slice[0]).unwrap();
        } else {
            string.push('[');
            write!(&mut string, "{}", slice[0]).unwrap();
            for i in slice.get(1..).unwrap() {
                write!(&mut string, ",{}", i).unwrap();
            }
            string.push(']');
        }
        string
    }
}
