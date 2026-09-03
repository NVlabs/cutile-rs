// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-side conversions for narrow floating-point storage formats.
//!
//! The conversion methods operate on the raw E4M3FN, E5M2, E2M1FN, and
//! E8M0FNU encodings used by the corresponding storage types. Every finite
//! narrow value is exactly representable in `f32`. Narrowing to E4M3FN, E5M2,
//! and E2M1FN uses round-to-nearest, ties-to-even; overflow and NaN behavior is
//! documented on each method.

use crate::dtype::{f4e2m1fn, f8e4m3fn, f8e5m2, f8e8m0fnu};

/// Interpretation of bit patterns with an all-ones exponent.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Specials {
    /// IEEE-like: all-ones exponent is infinity (mantissa 0) or NaN.
    InfAndNan,
    /// All-ones exponent with an all-ones mantissa is the only NaN; there are
    /// no infinities, so that exponent otherwise holds finite values.
    NanOnAllOnesMantissa,
    /// No infinities and no NaN; every bit pattern is a finite number.
    Finite,
}

/// Layout of a narrow float format.
struct Format {
    exp_bits: u32,
    mant_bits: u32,
    bias: i32,
    specials: Specials,
}

const E4M3FN: Format = Format {
    exp_bits: 4,
    mant_bits: 3,
    bias: 7,
    specials: Specials::NanOnAllOnesMantissa,
};

const E5M2: Format = Format {
    exp_bits: 5,
    mant_bits: 2,
    bias: 15,
    specials: Specials::InfAndNan,
};

const E2M1FN: Format = Format {
    exp_bits: 2,
    mant_bits: 1,
    bias: 1,
    specials: Specials::Finite,
};

impl Format {
    const fn max_exp(&self) -> i32 {
        (1i32 << self.exp_bits) - 1
    }

    const fn mant_mask(&self) -> u8 {
        (1u8 << self.mant_bits) - 1
    }

    const fn sign_shift(&self) -> u32 {
        self.exp_bits + self.mant_bits
    }

    /// The exponent and mantissa of the largest finite magnitude.
    const fn max_finite(&self) -> (i32, u8) {
        match self.specials {
            // The all-ones exponent is reserved for inf/NaN.
            Specials::InfAndNan => (self.max_exp() - 1, self.mant_mask()),
            // Only the all-ones mantissa is reserved, so max finite is one below.
            Specials::NanOnAllOnesMantissa => (self.max_exp(), self.mant_mask() - 1),
            Specials::Finite => (self.max_exp(), self.mant_mask()),
        }
    }
}

/// Returns `2^exp` as an `f32` for every exactly representable power of two.
#[inline]
fn exp2i(exp: i32) -> f32 {
    debug_assert!(
        (-149..=127).contains(&exp),
        "exponent {exp} out of f32 range"
    );
    if exp >= -126 {
        f32::from_bits(((exp + 127) as u32) << 23)
    } else {
        f32::from_bits(1 << (exp + 149))
    }
}

/// Decodes a narrow float bit pattern to `f32`.
///
/// Finite values are represented exactly. NaN encodings produce `f32::NAN`
/// without preserving their sign or payload.
#[inline]
fn decode(bits: u8, fmt: &Format) -> f32 {
    let negative = (bits >> fmt.sign_shift()) & 1 == 1;
    let exp = ((bits >> fmt.mant_bits) & ((1u8 << fmt.exp_bits) - 1)) as i32;
    let mant = bits & fmt.mant_mask();
    let mant_scale = (1u32 << fmt.mant_bits) as f32;

    let magnitude = if exp == fmt.max_exp() {
        match fmt.specials {
            Specials::InfAndNan => {
                if mant == 0 {
                    f32::INFINITY
                } else {
                    return f32::NAN;
                }
            }
            Specials::NanOnAllOnesMantissa => {
                if mant == fmt.mant_mask() {
                    return f32::NAN;
                }
                (1.0 + mant as f32 / mant_scale) * exp2i(exp - fmt.bias)
            }
            Specials::Finite => (1.0 + mant as f32 / mant_scale) * exp2i(exp - fmt.bias),
        }
    } else if exp == 0 {
        // Subnormal: no implicit leading one.
        mant as f32 * exp2i(1 - fmt.bias - fmt.mant_bits as i32)
    } else {
        (1.0 + mant as f32 / mant_scale) * exp2i(exp - fmt.bias)
    };

    if negative {
        -magnitude
    } else {
        magnitude
    }
}

/// Encodes an `f32` into a narrow float bit pattern, rounding to nearest with
/// ties to even.
#[inline]
fn encode(value: f32, fmt: &Format) -> u8 {
    let sign = if value.is_sign_negative() {
        1u8 << fmt.sign_shift()
    } else {
        0
    };
    let (max_finite_exp, max_finite_mant) = fmt.max_finite();
    let saturated = sign | ((max_finite_exp as u8) << fmt.mant_bits) | max_finite_mant;

    if value.is_nan() {
        return match fmt.specials {
            Specials::InfAndNan | Specials::NanOnAllOnesMantissa => {
                ((fmt.max_exp() as u8) << fmt.mant_bits) | fmt.mant_mask()
            }
            // Formats without a NaN encoding use positive max as the canonical result.
            Specials::Finite => saturated & !sign,
        };
    }
    if value.is_infinite() {
        return match fmt.specials {
            Specials::InfAndNan => sign | ((fmt.max_exp() as u8) << fmt.mant_bits),
            _ => saturated,
        };
    }

    let bits = value.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    let f32_mant = bits & 0x7F_FFFF;

    if f32_exp == 0 && f32_mant == 0 {
        return sign; // signed zero
    }

    // Normalize, so f32 subnormals are handled on the same path as normals.
    let (unbiased, mant24) = if f32_exp == 0 {
        let shift = f32_mant.leading_zeros() - 8;
        (1 - 127 - shift as i32, (f32_mant << shift) & 0x7F_FFFF)
    } else {
        (f32_exp - 127, f32_mant)
    };

    let target_exp = unbiased + fmt.bias;
    // Implicit leading one restored at bit 23.
    let full = (1u32 << 23) | mant24;

    // How far right to shift to land the mantissa in `mant_bits`. Subnormal
    // results clamp the exponent to zero and absorb the difference as extra
    // shift, which is what makes gradual underflow work.
    let (mut out_exp, shift) = if target_exp <= 0 {
        (0i32, 23 - fmt.mant_bits as i32 + (1 - target_exp))
    } else {
        (target_exp, 23 - fmt.mant_bits as i32)
    };

    // Shifting out more bits than the value has leaves nothing but a rounding
    // decision against the smallest subnormal.
    if shift > 24 {
        return sign;
    }

    let mut out_mant = full >> shift;
    let dropped = full & ((1u32 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    if dropped > halfway || (dropped == halfway && out_mant & 1 == 1) {
        out_mant += 1;
    }

    if out_exp == 0 {
        // Rounding may have carried a subnormal up into the smallest normal.
        if out_mant >= (1u32 << fmt.mant_bits) {
            out_exp = 1;
            out_mant &= fmt.mant_mask() as u32;
        }
    } else {
        // Rounding may have carried into the next binade.
        if out_mant >= (1u32 << (fmt.mant_bits + 1)) {
            out_exp += 1;
            out_mant >>= 1;
        }
        out_mant &= fmt.mant_mask() as u32;
    }

    // Overflow, including overflow produced by the rounding above.
    if out_exp > max_finite_exp || (out_exp == max_finite_exp && out_mant > max_finite_mant as u32)
    {
        return match fmt.specials {
            Specials::InfAndNan => sign | ((fmt.max_exp() as u8) << fmt.mant_bits),
            _ => saturated,
        };
    }

    sign | ((out_exp as u8) << fmt.mant_bits) | out_mant as u8
}

impl f8e4m3fn {
    /// Converts this E4M3FN encoding to `f32`.
    ///
    /// Every finite E4M3FN value is represented exactly.
    #[inline]
    pub fn to_f32(self) -> f32 {
        decode(self.0, &E4M3FN)
    }

    /// Converts `value` to E4M3FN using round-to-nearest, ties-to-even.
    ///
    /// This format has no infinities, so infinite and out-of-range inputs
    /// saturate to ±448 rather than becoming NaN. Every NaN input produces the
    /// canonical encoding `0x7F`.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        Self(encode(value, &E4M3FN))
    }
}

impl f8e5m2 {
    /// Converts this E5M2 encoding to `f32`.
    ///
    /// Every finite E5M2 value is represented exactly.
    #[inline]
    pub fn to_f32(self) -> f32 {
        decode(self.0, &E5M2)
    }

    /// Converts `value` to E5M2 using round-to-nearest, ties-to-even.
    ///
    /// This format is IEEE-like: finite overflow produces infinity. Every NaN
    /// input produces the canonical positive encoding `0x7F`.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        Self(encode(value, &E5M2))
    }
}

impl f4e2m1fn {
    /// Converts this E2M1FN encoding to `f32`.
    ///
    /// The format represents exactly ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    #[inline]
    pub fn to_f32(self) -> f32 {
        decode(self.0 & 0x0F, &E2M1FN)
    }

    /// Converts `value` to E2M1FN using round-to-nearest, ties-to-even.
    ///
    /// This format has neither infinities nor NaN. Infinities and out-of-range
    /// inputs saturate to ±6; every NaN input produces positive 6.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        Self(encode(value, &E2M1FN))
    }
}

impl f8e8m0fnu {
    /// The exponent bias: a stored byte `b` denotes `2^(b - 127)`.
    pub const BIAS: i32 = 127;

    /// The bit pattern denoting NaN. This format has no infinities and no zero.
    pub const NAN: Self = Self(0xFF);

    /// Converts to `f32`, returning NaN for the reserved pattern.
    ///
    /// This format is a bare power of two with no sign or mantissa and is used
    /// for per-block MX tensor scales.
    #[inline]
    pub fn to_f32(self) -> f32 {
        if self.0 == 0xFF {
            return f32::NAN;
        }
        exp2i(self.0 as i32 - Self::BIAS)
    }

    /// Returns the exponent this scale denotes, so `to_f32() == 2^exponent()`.
    #[inline]
    pub fn exponent(self) -> Option<i32> {
        if self.0 == 0xFF {
            None
        } else {
            Some(self.0 as i32 - Self::BIAS)
        }
    }

    /// Rounds `magnitude` upward to an E8M0 scale.
    ///
    /// For a positive finite input in the representable range, the result is
    /// `2^ceil(log2(magnitude))`; exact powers of two are unchanged. This upward
    /// rounding is suitable for block scaling because the encoded scale does not
    /// undershoot the requested magnitude.
    ///
    /// Positive values below `2^-127` return `2^-127`. Values above `2^127` and
    /// positive infinity return `2^127`. NaN returns [`Self::NAN`], and
    /// non-positive inputs return `2^-127`.
    #[inline]
    pub fn scale_covering(magnitude: f32) -> Self {
        if magnitude.is_nan() {
            return Self::NAN;
        }
        if magnitude <= 0.0 {
            return Self(0);
        }
        if magnitude.is_infinite() {
            return Self(0xFE);
        }
        let bits = magnitude.to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32;
        let f32_mant = bits & 0x7F_FFFF;
        let exponent = if f32_exp == 0 {
            // A subnormal is `mantissa * 2^-149`. Its highest set bit gives
            // floor(log2), and a second set bit means the ceiling is next.
            let floor = f32_mant.ilog2() as i32 - 149;
            floor + i32::from(!f32_mant.is_power_of_two())
        } else if f32_mant == 0 {
            // Exactly a power of two: it covers itself.
            f32_exp - 127
        } else {
            f32_exp - 127 + 1
        };
        Self((exponent.clamp(-127, 127) + Self::BIAS) as u8)
    }
}

macro_rules! impl_widen_to_f32 {
    ($($narrow:ty),+ $(,)?) => {
        $(
            impl From<$narrow> for f32 {
                #[inline]
                fn from(value: $narrow) -> Self {
                    value.to_f32()
                }
            }
        )+
    };
}

impl_widen_to_f32!(f8e4m3fn, f8e5m2, f4e2m1fn, f8e8m0fnu);

#[cfg(test)]
mod tests {
    use super::*;

    /// Every bit pattern of a 1+exp+mant format, given its total width.
    fn all_patterns(width: u32) -> impl Iterator<Item = u8> {
        0..=((1u16 << width) - 1) as u8
    }

    fn exact_power_of_two(exponent: i32) -> f32 {
        assert!((-149..=127).contains(&exponent));
        if exponent >= -126 {
            f32::from_bits(((exponent + 127) as u32) << 23)
        } else {
            f32::from_bits(1 << (exponent + 149))
        }
    }

    fn assert_rne_boundaries(
        positive_patterns: impl IntoIterator<Item = u8>,
        decode: impl Fn(u8) -> f32,
        encode: impl Fn(f32) -> u8,
        sign_bit: u8,
    ) {
        let patterns: Vec<_> = positive_patterns.into_iter().collect();
        for pair in patterns.windows(2) {
            let [lower_bits, upper_bits] = pair else {
                unreachable!()
            };
            let lower = decode(*lower_bits);
            let upper = decode(*upper_bits);
            let midpoint = (lower + upper) / 2.0;
            let below_midpoint = f32::from_bits(midpoint.to_bits() - 1);
            let above_midpoint = f32::from_bits(midpoint.to_bits() + 1);
            let tie_winner = if lower_bits & 1 == 0 {
                *lower_bits
            } else {
                *upper_bits
            };

            assert_eq!(encode(below_midpoint), *lower_bits);
            assert_eq!(encode(midpoint), tie_winner);
            assert_eq!(encode(above_midpoint), *upper_bits);

            assert_eq!(encode(-below_midpoint), sign_bit | *lower_bits);
            assert_eq!(encode(-midpoint), sign_bit | tie_winner);
            assert_eq!(encode(-above_midpoint), sign_bit | *upper_bits);
        }
    }

    #[test]
    fn e4m3fn_round_trips_every_bit_pattern() {
        for bits in all_patterns(8) {
            let value = f8e4m3fn(bits).to_f32();
            if value.is_nan() {
                continue; // NaN has several encodings; not a round-trip target
            }
            let back = f8e4m3fn::from_f32(value);
            assert_eq!(
                back.0, bits,
                "0x{bits:02X} decoded to {value} re-encoded to 0x{:02X}",
                back.0
            );
        }
    }

    #[test]
    fn e5m2_round_trips_every_bit_pattern() {
        for bits in all_patterns(8) {
            let value = f8e5m2(bits).to_f32();
            if value.is_nan() {
                continue;
            }
            let back = f8e5m2::from_f32(value);
            assert_eq!(
                back.0, bits,
                "0x{bits:02X} decoded to {value} re-encoded to 0x{:02X}",
                back.0
            );
        }
    }

    #[test]
    fn e2m1fn_round_trips_every_bit_pattern() {
        for bits in all_patterns(4) {
            let value = f4e2m1fn(bits).to_f32();
            assert!(
                value.is_finite(),
                "FP4 has no non-finite values, got {value}"
            );
            let back = f4e2m1fn::from_f32(value);
            assert_eq!(back.0, bits, "0x{bits:X} decoded to {value}");
        }
    }

    #[test]
    fn e2m1fn_holds_exactly_the_documented_value_set() {
        let mut positives: Vec<f32> = (0..8u8).map(|b| f4e2m1fn(b).to_f32()).collect();
        positives.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(positives, vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn format_limits_match_the_specifications() {
        assert_eq!(f8e4m3fn(0x7E).to_f32(), 448.0);
        assert!(f8e4m3fn(0x7F).to_f32().is_nan());
        assert_eq!(f8e5m2(0x7B).to_f32(), 57344.0);
        assert!(f8e5m2(0x7C).to_f32().is_infinite());
        assert_eq!(f4e2m1fn(0x7).to_f32(), 6.0);
    }

    #[test]
    fn widening_implements_the_standard_from_conversion() {
        assert_eq!(f32::from(f8e4m3fn(0x38)), 1.0);
        assert_eq!(f32::from(f8e5m2(0x3C)), 1.0);
        assert_eq!(f32::from(f4e2m1fn(0x02)), 1.0);
        assert_eq!(f32::from(f8e8m0fnu(0x7F)), 1.0);
    }

    #[test]
    fn out_of_range_saturates_in_formats_without_infinity() {
        assert_eq!(f8e4m3fn::from_f32(1.0e30).0, 0x7E); // +448
        assert_eq!(f8e4m3fn::from_f32(-1.0e30).0, 0xFE); // -448
        assert_eq!(f8e4m3fn::from_f32(f32::INFINITY).0, 0x7E);
        assert_eq!(f4e2m1fn::from_f32(1000.0).0, 0x7); // +6
        assert_eq!(f4e2m1fn::from_f32(f32::NEG_INFINITY).0, 0xF); // -6
    }

    #[test]
    fn out_of_range_becomes_infinity_in_ieee_like_format() {
        assert_eq!(f8e5m2::from_f32(1.0e30).0, 0x7C);
        assert!(f8e5m2::from_f32(f32::INFINITY).to_f32().is_infinite());
        assert!(f8e5m2::from_f32(f32::NAN).to_f32().is_nan());
    }

    #[test]
    fn nan_inputs_use_cuda_canonical_encodings() {
        // Cover positive/negative, quiet/signaling, and minimal/maximal
        // payloads. CUDA discards both the payload and the input sign.
        let nan_patterns = [
            0x7F80_0001,
            0x7FC0_0000,
            0x7FFF_FFFF,
            0xFF80_0001,
            0xFFC0_0000,
            0xFFFF_FFFF,
        ];
        for bits in nan_patterns {
            let value = f32::from_bits(bits);
            assert!(value.is_nan());
            assert_eq!(f8e4m3fn::from_f32(value).0, 0x7F);
            assert_eq!(f8e5m2::from_f32(value).0, 0x7F);
            assert_eq!(f4e2m1fn::from_f32(value).0, 0x07);
        }
    }

    #[test]
    fn ties_round_to_even_not_away_from_zero() {
        // E4M3FN near 1.0: representable neighbours are 1.0 (mant 000) and
        // 1.125 (mant 001), so 1.0625 is an exact tie. Even mantissa wins.
        assert_eq!(f8e4m3fn::from_f32(1.0625).0, f8e4m3fn::from_f32(1.0).0);
        // 1.1875 ties between 1.125 (mant 001) and 1.25 (mant 010); even wins.
        assert_eq!(f8e4m3fn::from_f32(1.1875).0, f8e4m3fn::from_f32(1.25).0);
    }

    #[test]
    fn every_finite_rounding_boundary_uses_ties_to_even() {
        assert_rne_boundaries(
            0x00..=0x7E,
            |bits| f8e4m3fn(bits).to_f32(),
            |value| f8e4m3fn::from_f32(value).0,
            0x80,
        );
        assert_rne_boundaries(
            0x00..=0x7B,
            |bits| f8e5m2(bits).to_f32(),
            |value| f8e5m2::from_f32(value).0,
            0x80,
        );
        assert_rne_boundaries(
            0x00..=0x07,
            |bits| f4e2m1fn(bits).to_f32(),
            |value| f4e2m1fn::from_f32(value).0,
            0x08,
        );

        // E5M2's final rounding boundary is between its largest finite value
        // and infinity. The exact tie rounds to the even infinity encoding.
        let overflow_midpoint = 61_440.0f32;
        assert_eq!(
            f8e5m2::from_f32(f32::from_bits(overflow_midpoint.to_bits() - 1)).0,
            0x7B
        );
        assert_eq!(f8e5m2::from_f32(overflow_midpoint).0, 0x7C);
    }

    #[test]
    fn signed_zero_is_preserved() {
        assert_eq!(f8e4m3fn::from_f32(0.0).0, 0x00);
        assert_eq!(f8e4m3fn::from_f32(-0.0).0, 0x80);
        assert_eq!(f8e5m2::from_f32(-0.0).0, 0x80);
        assert_eq!(f4e2m1fn::from_f32(-0.0).0, 0x8);
    }

    #[test]
    fn subnormals_are_gradual_not_flushed() {
        // E4M3FN smallest subnormal is 2^-9; smallest normal is 2^-6.
        let smallest_subnormal = exact_power_of_two(-9);
        assert_eq!(f8e4m3fn::from_f32(smallest_subnormal).0, 0x01);
        assert_eq!(f8e4m3fn(0x01).to_f32(), smallest_subnormal);
        // Half of it rounds to even, which is zero.
        assert_eq!(f8e4m3fn::from_f32(smallest_subnormal / 2.0).0, 0x00);
        // Just above half rounds up to the subnormal.
        assert_eq!(f8e4m3fn::from_f32(smallest_subnormal * 0.75).0, 0x01);
    }

    #[test]
    fn tiny_inputs_underflow_to_signed_zero() {
        assert_eq!(f8e4m3fn::from_f32(1.0e-30).0, 0x00);
        assert_eq!(f8e4m3fn::from_f32(-1.0e-30).0, 0x80);
        // The smallest f32 subnormal rounds to zero.
        assert_eq!(f8e4m3fn::from_f32(f32::from_bits(1)).0, 0x00);
    }

    #[test]
    fn e8m0_scale_is_a_bare_power_of_two() {
        assert_eq!(f8e8m0fnu(0).to_f32(), f32::from_bits(1 << 22));
        assert_eq!(f8e8m0fnu(0).to_f32(), exact_power_of_two(-127));
        assert_eq!(f8e8m0fnu(127).to_f32(), 1.0);
        assert_eq!(f8e8m0fnu(128).to_f32(), 2.0);
        assert_eq!(f8e8m0fnu(126).to_f32(), 0.5);
        assert_eq!(f8e8m0fnu(0).exponent(), Some(-127));
        assert_eq!(f8e8m0fnu(127).exponent(), Some(0));
        assert!(f8e8m0fnu::NAN.to_f32().is_nan());
        assert_eq!(f8e8m0fnu::NAN.exponent(), None);
    }

    #[test]
    fn e8m0_decodes_every_non_nan_pattern() {
        for bits in 0..=0xFE {
            let exponent = bits as i32 - f8e8m0fnu::BIAS;
            let expected = exact_power_of_two(exponent);
            assert_eq!(
                f8e8m0fnu(bits).to_f32().to_bits(),
                expected.to_bits(),
                "byte 0x{bits:02X} should decode to 2^{exponent}"
            );
        }
    }

    #[test]
    fn scale_covering_never_rounds_down() {
        // A covering scale is never smaller than an in-range magnitude.
        let mut value = 1.0e-30f32;
        while value < 1.0e30 {
            let scale = f8e8m0fnu::scale_covering(value).to_f32();
            assert!(
                scale >= value,
                "scale {scale} does not cover {value} - elements would saturate"
            );
            value *= 1.7;
        }
    }

    #[test]
    fn scale_covering_is_minimal_across_f32_binades() {
        // Exercise every f32 exponent class and the mantissa shapes that
        // distinguish exact powers, near-powers, and binade endpoints.
        let mantissas = [0, 1, 2, 3, 0x3F_FFFF, 0x40_0000, 0x40_0001, 0x7F_FFFF];
        let largest_scale = exact_power_of_two(127);
        let mut previous = 0;

        for f32_exp in 0..=254u32 {
            for mantissa in mantissas {
                let value = f32::from_bits((f32_exp << 23) | mantissa);
                if value == 0.0 {
                    continue;
                }

                let scale = f8e8m0fnu::scale_covering(value);
                assert!(scale.0 >= previous, "scale decreased at {value:e}");
                previous = scale.0;

                let decoded = scale.to_f32();
                if value <= largest_scale {
                    assert!(decoded >= value, "{decoded:e} does not cover {value:e}");
                } else {
                    assert_eq!(decoded, largest_scale);
                }

                if let Some(exponent) = scale.exponent().filter(|&exp| exp > -127) {
                    assert!(
                        exact_power_of_two(exponent - 1) < value,
                        "{decoded:e} is not the minimal scale covering {value:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn scale_covering_is_exact_on_powers_of_two() {
        // A power of two covers itself; rounding up here would waste a whole
        // exponent of range on every block whose max is already a power of two.
        for exp in -100..=100 {
            let value = exact_power_of_two(exp);
            assert_eq!(
                f8e8m0fnu::scale_covering(value).exponent(),
                Some(exp),
                "2^{exp} should map to itself"
            );
        }
    }

    #[test]
    fn scale_covering_rounds_up_off_powers_of_two() {
        assert_eq!(f8e8m0fnu::scale_covering(1.5).exponent(), Some(1)); // 2.0
        assert_eq!(f8e8m0fnu::scale_covering(3.0).exponent(), Some(2)); // 4.0
        assert_eq!(f8e8m0fnu::scale_covering(0.6).exponent(), Some(0)); // 1.0
    }

    #[test]
    fn scale_covering_handles_degenerate_inputs() {
        assert_eq!(f8e8m0fnu::scale_covering(0.0).0, 0);
        assert_eq!(f8e8m0fnu::scale_covering(-0.0).0, 0);
        assert_eq!(f8e8m0fnu::scale_covering(-5.0).0, 0);
        assert_eq!(f8e8m0fnu::scale_covering(f32::NAN).0, 0xFF);
        assert_eq!(f8e8m0fnu::scale_covering(f32::INFINITY).0, 0xFE);
        assert_eq!(f8e8m0fnu::scale_covering(f32::NEG_INFINITY).0, 0);
    }

    #[test]
    fn scale_covering_clamps_rather_than_wrapping() {
        let largest_scale = exact_power_of_two(127);
        assert_eq!(
            f8e8m0fnu::scale_covering(largest_scale).to_f32(),
            largest_scale
        );

        // Above the format's range, the result remains at the largest scale and
        // no longer covers the input.
        let huge = f8e8m0fnu::scale_covering(f32::MAX);
        assert_eq!(huge.exponent(), Some(127));
        assert_eq!(huge.to_f32(), largest_scale);
        assert!(huge.to_f32() < f32::MAX);

        let tiny = f8e8m0fnu::scale_covering(f32::from_bits(1));
        assert_eq!(tiny.exponent(), Some(-127));
    }
}
