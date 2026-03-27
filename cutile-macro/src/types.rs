/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Type system and const generic array handling.
//!
//! This module implements the type system for cuTile Rust's rank-polymorphic types.
//! It handles the expansion and tracking of variadic types that use const generic
//! array parameters.
//!
//! ## Purpose
//!
//! This module exists to:
//!
//! 1. **Generate concrete structs** for types with const generic arrays
//! 2. **Expand function definitions** based on const generic array parameters
//! 3. **Rewrite function calls** to use the correct rank-specialized versions
//! 4. **Track type information** throughout the macro expansion process
//!
//! ## Variadic Type System
//!
//! cuTile Rust uses a variadic type system where types like `Tile`, `Tensor`, and
//! `Partition` exist in multiple rank-specific versions:
//!
//! ```rust,ignore
//! // Generic definition (not actual Rust):
//! struct Tile<E, const D: [i32; N]> { }
//!
//! // Concrete instantiations:
//! struct Tile_1<E, const D: [i32; 1]> { } // 1D
//! struct Tile_2<E, const D: [i32; 2]> { } // 2D
//! struct Tile_3<E, const D: [i32; 3]> { } // 3D
//! struct Tile_4<E, const D: [i32; 4]> { } // 4D
//! ```
//!
//! ## Type Registry
//!
//! The module maintains a registry of known variadic types (`VARIADIC_TYPES`)
//! with metadata about each type including:
//!
//! - Const generic array parameter names
//! - Dimension types (static vs mixed)
//! - Index types
//!
//! ## Const Generic Arrays (CGAs)
//!
//! CGAs are array-valued const generic parameters like `const S: [i32; N]`. They
//! represent compile-time shape information:
//!
//! - **Static dimensions**: All dimensions are compile-time constants (e.g., `{[128, 64]}`)
//! - **Mixed dimensions**: Some dimensions may be dynamic -1 (e.g., `{[-1, 64]}`)
//!
//! ## Type Inference
//!
//! The type system infers concrete types from usage:
//!
//! ```rust,ignore
//! // From this:
//! let tile: Tile<f32, {[128, 64]}> = ...;
//!
//! // Infers: Tile_2<f32, 128, 64>
//! ```

use phf::phf_map;
use proc_macro2::Span;
use std::collections::HashMap;

use crate::error::{Error, SpannedError};

/// Generates a suffix string for a variadic type based on its rank.
///
/// For example, `[2]` becomes `"2"`, `[2, 3]` becomes `"2_3"`.
/// Generates a type name suffix based on the variadic rank vector.
///
/// For variadic types (like `Tile`, `Tensor`, `Shape`), this appends the rank
/// to the type name. For example, a 2D tile becomes `Tile_2`.
///
/// ## Parameters
///
/// - `n`: Array of ranks for each const-generic array parameter
///
/// ## Returns
///
/// A string suffix like `"_2"` for 2D types, `"_3"` for 3D, etc.
/// Returns `"_0"` for scalar (0D) types.
///
/// ## Example
///
/// ```rust,ignore
/// let suffix = get_variadic_type_suffix(&[2]); // Returns "_2"
/// let type_name = format!("Tile{}", suffix);   // "Tile_2"
/// ```
/// Generates a rank suffix string from an array of dimension counts (e.g., `[2]` → `"2"`, `[2, 3]` → `"2_3"`).
pub fn get_variadic_type_suffix(n: &[u32]) -> String {
    n.iter()
        .map(|v| v.to_string())
        .collect::<Vec<String>>()
        .join("_")
}

/// Generates the concrete name for a variadic type instance.
///
/// Appends the rank suffix to the base type name.
///
/// ## Examples
///
/// ```rust,ignore
/// concrete_name("Tile", &[2]) // "Tile_2"
/// concrete_name("Shape", &[3]) // "Shape_3"
/// ```
/// Generates a concrete type name for a variadic type.
///
/// Combines a base type name with the variadic suffix to produce the full
/// concrete type name used in generated code.
///
/// ## Parameters
///
/// - `name`: Base type name (e.g., "Tile", "Tensor", "Shape")
/// - `n`: Array of ranks for each const-generic array parameter
///
/// ## Returns
///
/// The concrete type name (e.g., "Tile_2", "Tensor_3", "Shape_0")
///
/// ## Example
///
/// ```rust,ignore
/// let name = concrete_name("Tile", &[2]); // Returns "Tile_2"
/// ```
/// Builds a concrete type name by appending the rank suffix (e.g., `"Tile"` + `[2]` → `"Tile_2"`).
pub fn concrete_name(name: &str, n: &[u32]) -> String {
    format!("{}_{}", name, get_variadic_type_suffix(n))
}

/// Classifies how dimensions are specified in a const generic array.
/// Describes whether a const-generic array parameter allows dynamic dimensions.
///
/// In cutile, shapes can be either fully static (all dimensions known at
/// compile-time) or mixed (some dimensions can be `-1` for runtime-determined sizes).
///
/// ## Variants
///
/// - **`Static`**: All dimensions must be compile-time constants (no `-1` allowed)
///   - Example: `Tile<f32, {[128, 64]}>` - shape known at compile-time
/// - **`Mixed`**: Dimensions can be static or dynamic (`-1`)
///   - Example: `Tensor<f32, {[-1, 64]}>` - first dimension determined at runtime
///
/// Whether a const-generic array parameter allows dynamic (`-1`) dimensions.
#[derive(Debug, Clone, PartialEq)]
pub enum DimType {
    /// All dimensions must be compile-time constants (no `-1` allowed)
    Static,
    /// Dimensions can be static or dynamic (`-1`)
    Mixed,
}

/// Metadata about a variadic type.
///
/// Contains information needed to generate and expand rank-specific versions
/// of a variadic type.
#[derive(Debug, Clone)]
/// Metadata describing a variadic type's structure and constraints.
///
/// Variadic types in cutile (like `Tile`, `Tensor`, `Shape`) support multiple
/// ranks (0D through 4D). This struct describes the parameters and constraints for
/// a variadic type family.
///
/// ## Fields
///
/// - **`name`**: Base name of the type (e.g., "Tile", "Tensor", "Shape")
/// - **`cga_names`**: Names of const generic array (CGA) parameters
///   - Example: `["D"]` for `Tile<E, const D: [i32; N]>`
/// - **`cga_dim_types`**: Dimension type restrictions for each CGA
///   - Controls whether `-1` (dynamic dimensions) are allowed
/// - **`cga_index_types`**: Element types for array indices (typically "i32")
///
/// ## Example
///
/// For `Tile<E, const D: [i32; N]>`:
/// ```rust,ignore
/// VariadicTypeData {
///     name: "Tile",
///     cga_names: &["D"],
///     cga_dim_types: &[DimType::Static],  // Tiles must have static shapes
///     cga_index_types: &["i32"],
/// }
/// ```
/// Metadata describing a variadic type's name, CGA parameters, and dimension constraints.
pub struct VariadicTypeData {
    /// Base name of the type (e.g., "Tile", "Tensor")
    pub name: &'static str,
    /// Names of const generic array parameters
    pub cga_names: &'static [&'static str],
    /// Dimension type restrictions for each CGA
    pub cga_dim_types: &'static [DimType],
    /// Element types for array indices (e.g., "i32")
    pub cga_index_types: &'static [&'static str],
}

impl VariadicTypeData {
    /// Returns the concrete name for this type at a specific rank.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let vtd = get_variadic_type_data("Tile").unwrap();
    /// assert_eq!(vtd.concrete_name(&vec![2]), "Tile_2");
    /// ```
    pub fn concrete_name(&self, n: &[u32]) -> String {
        concrete_name(self.name, n)
    }

    /// Returns the concrete name from a const generic array type instance.
    pub fn concrete_name_from_cga_type(&self, cga_type: &ConstGenericArrayType) -> String {
        // Get the name of this vtd from a ConstGenericArrayType instance.
        self.concrete_name(&cga_type.n)
    }

    /// Returns the number of const generic arrays this type has.
    pub fn num_cgas(&self) -> u32 {
        self.cga_names.len() as u32
    }

    /// Creates an iterator over all rank combinations for this type.
    pub fn iter(&self, n_vec: &[u32]) -> ConstGenericArrayTypeIterator {
        ConstGenericArrayTypeIterator::new(n_vec)
    }
}

/// Registry of all variadic types in the DSL.
///
/// This maps type names to their metadata, enabling the macro system to correctly
/// expand and specialize variadic types. Each entry defines how a type should be
/// instantiated for different ranks.
static VARIADIC_TYPES: phf::Map<&'static str, VariadicTypeData> = phf_map! {
    "Array" => VariadicTypeData {name: "Array", cga_index_types: &["i32"], cga_names: &["D"], cga_dim_types: &[DimType::Mixed]},
    "Shape" => VariadicTypeData {name: "Shape", cga_index_types: &["i32"], cga_names: &["D"], cga_dim_types: &[DimType::Mixed]},
    "PointerTile" => VariadicTypeData {name: "PointerTile", cga_index_types: &["i32"], cga_names: &["D"], cga_dim_types: &[DimType::Static]},
    "Tensor" => VariadicTypeData {name: "Tensor", cga_index_types: &["i32"], cga_names: &["D"], cga_dim_types: &[DimType::Mixed]},
    "Partition" => VariadicTypeData {name: "Partition", cga_index_types: &["i32"], cga_names: &["D"], cga_dim_types: &[DimType::Static]},
    "PartitionMut" => VariadicTypeData {name: "PartitionMut", cga_index_types: &["i32"], cga_names: &["D"], cga_dim_types: &[DimType::Static]},
    "Tile" => VariadicTypeData {name: "Tile", cga_index_types: &["i32"], cga_names: &["D"], cga_dim_types: &[DimType::Static]},
    "BroadcastScalar" => VariadicTypeData {name: "BroadcastScalar", cga_index_types: &["i32"], cga_names: &["D"], cga_dim_types: &[DimType::Static]},
};

/// Looks up variadic type data for trait method calls.
///
/// Special handling for methods like `broadcast` that work on primitive types
/// through trait implementations.
// Unfortunately required.
/// Retrieves variadic type data for trait methods.
///
/// Some traits (like `BroadcastScalar`) are implemented variadically across multiple
/// ranks. This function looks up the type data for the receiver type of a method.
///
/// ## Parameters
///
/// - `maybe_primitive`: The receiver type name (e.g., "i32", "f32")
/// - `method_name`: The method name (e.g., "broadcast")
///
/// ## Returns
///
/// `Some(VariadicTypeData)` if the method is variadic, `None` otherwise
///
/// ## Example
///
/// ```rust,ignore
/// // For: impl BroadcastScalar<E, D> for E { fn broadcast(...) }
/// let data = get_variadic_trait_type_data("f32", "broadcast");
/// // Returns VariadicTypeData for the broadcast method
/// ```
/// Looks up variadic type data for a trait method on a primitive receiver type.
pub fn get_variadic_trait_type_data(
    maybe_primitive: &str,
    method_name: &str,
) -> Option<VariadicTypeData> {
    match (maybe_primitive, method_name) {
        // This is the only reason this works.
        ("T", "broadcast") => get_variadic_type_data("BroadcastScalar"),
        ("f32", "broadcast") => get_variadic_type_data("BroadcastScalar"),
        ("i32", "broadcast") => get_variadic_type_data("BroadcastScalar"),
        ("u32", "broadcast") => get_variadic_type_data("BroadcastScalar"),
        ("bool", "broadcast") => get_variadic_type_data("BroadcastScalar"),
        _ => None,
    }
}

/// Looks up metadata for a variadic type by name.
///
/// Returns `None` if the type is not in the registry or is not variadic.
///
/// ## Examples
///
/// ```rust,ignore
/// let tile_data = get_variadic_type_data("Tile");
/// assert!(tile_data.is_some());
///
/// let not_variadic = get_variadic_type_data("Vec");
/// assert!(not_variadic.is_none());
/// ```
/// Retrieves variadic type data for a type by name.
///
/// Looks up the metadata for variadic types like `Tile`, `Tensor`, `Shape`, etc.
/// This is used during macro expansion to generate concrete types for each rank.
///
/// ## Parameters
///
/// - `type_name`: The type name to look up (e.g., "Tile", "Tensor", "Shape")
///
/// ## Returns
///
/// `Some(VariadicTypeData)` if the type is variadic, `None` otherwise
///
/// ## Supported Variadic Types
///
/// - **Tiles and Operations**: `Tile`, `PointerTile`
/// - **Memory Views**: `Tensor`, `Partition`, `PartitionMut`
/// - **Metadata**: `Shape`, `Array`
///
/// ## Example
///
/// ```rust,ignore
/// let data = get_variadic_type_data("Tile").unwrap();
/// assert_eq!(data.name, "Tile");
/// assert_eq!(data.cga_names, &["D"]);
/// ```
/// Returns metadata for a variadic type by name, or `None` if not variadic.
pub fn get_variadic_type_data(type_name: &str) -> Option<VariadicTypeData> {
    VARIADIC_TYPES.get(type_name).cloned()
}

/// Metadata about a variadic operation (function or method).
///
/// Describes how a variadic function should be expanded, including the mapping
/// between const generic array parameters and the types they appear in.
///
/// ## Fields
///
/// - `const_length_vars` - Names of the rank variables (e.g., `["N", "M"]`)
/// - `cga_map` - Maps const generic array (CGA) names to their rank variables
/// - `input_map` - Input parameters: `(arg_index, type_name, [cga_names])`
/// - `output_map` - MLIR operation output: `(type_name, [cga_names])`
/// - `return_type` - Rust function return type: `(type_name, [element_type, cga_names])`
///
/// ## Example
///
/// For a reshape operation:
/// ```rust,ignore
/// fn reshape<E: ElementType, const S: [i32; N], const R: [i32; M]>(
///     tile: Tile<E, S>,
///     shape: Shape<R>
/// ) -> Tile<E, R>
/// ```
///
/// ```rust,ignore
/// VariadicOpData {
///     const_length_vars: &["N", "M"],  // Two rank variables
///     cga_map: HashMap::from([
///         ("S", "N"),  // Input shape S has rank N
///         ("R", "M"),  // Output shape R has rank M
///     ]),
///     input_map: vec![
///         (0, "Tile", &["S"]),   // Arg 0: Tile<E, S> where S has rank N
///         (1, "Shape", &["R"]),  // Arg 1: Shape<R> where R has rank M
///     ],
///     output_map: ("Tile", &["R"]),         // MLIR output: tile with shape R
///     return_type: ("Tile", &["_", "R"]),   // Rust: Tile<_, R> (element inferred)
/// }
/// ```
///
/// This enables generating `reshape_1__2`, `reshape_2__3`, etc. for different rank combinations.
#[derive(Debug, Clone)]
/// Metadata describing a variadic operation's type signature.
///
/// Variadic operations (like `reshape`, `broadcast`, `reduce`) can operate on
/// tiles of different ranks. This struct describes how const-generic array
/// parameters map between inputs and outputs.
///
/// ## Fields
///
/// - **`const_length_vars`**: Names of const rank variables (e.g., `["N", "M"]`)
///   - These represent the rank of different CGA parameters
/// - **`cga_map`**: Maps CGA names to their rank variables
///   - Example: `{"S": "N", "R": "M"}` means shape S has rank N, shape R has rank M
/// - **`input_map`**: Type information for input parameters
///   - Each entry: (parameter_index, type_name, cga_names)
///   - Example: `(0, "Tile", ["S"])` means first param is `Tile<E, S>` with rank N
/// - **`output_map`**: Type information for the output/self type
///   - Example: `("Tile", ["R"])` for methods that return `Tile<E, R>`
/// - **`return_type`**: Type information for the return value
///   - Example: `("Tile", ["R"])` for functions returning `Tile<E, R>`
///
/// ## Example
///
/// For `fn reshape<const S: [i32; N], const R: [i32; M]>(self, ...) -> Tile<E, R>`:
/// ```rust,ignore
/// VariadicOpData {
///     const_length_vars: &["N", "M"],
///     cga_map: { "S": "N", "R": "M" },
///     input_map: vec![(1, "Shape", &["R"])],  // shape parameter
///     output_map: ("Tile", &["S"]),           // self is Tile<E, S>
///     return_type: ("Tile", &["R"]),          // returns Tile<E, R>
/// }
/// ```
/// Metadata describing a variadic operation's const-generic array signature and type mappings.
pub struct VariadicOpData {
    /// Names of const rank variables
    pub const_length_vars: &'static [&'static str], // [ length_var, ... ]
    /// Maps CGA names to their rank variables
    pub cga_map: HashMap<&'static str, &'static str>, // { cga_var: length_var, ... }
    /// Input parameter type information
    pub input_map: Vec<(usize, &'static str, &'static [&'static str])>, // [(type_name, [cga_var, ...]), ... ]
    /// Output type information
    pub output_map: (&'static str, &'static [&'static str]), // (type_name, [cga_var, ...])
    /// Return type information
    pub return_type: (&'static str, &'static [&'static str]),
}

/// Retrieves variadic operation data for a method.
///
/// Methods on variadic types (like `Tile::reshape`, `Tile::broadcast`) can have
/// complex type signatures involving multiple rank variables. This function looks
/// up the operation data for a specific method.
///
/// ## Parameters
///
/// - `vtd`: The variadic type data for the receiver type
/// - `method_name`: The method name (e.g., "reshape", "broadcast")
///
/// ## Returns
///
/// `Some((concrete_method_name, VariadicOpData))` if the method is variadic,
/// `None` otherwise. The concrete name includes rank suffixes (e.g., "reshape_2_3"
/// for reshaping from 2D to 3D).
///
/// ## Example
///
/// ```rust,ignore
/// let tile_data = get_variadic_type_data("Tile").unwrap();
/// let (name, op_data) = get_variadic_method_data(&tile_data, "reshape").unwrap();
/// // name might be "reshape" (base name)
/// // op_data describes the type signature
/// ```
/// Looks up variadic operation data for a method on a variadic type.
pub fn get_variadic_method_data(
    vtd: &VariadicTypeData,
    method_name: &str,
) -> Result<Option<(&'static str, VariadicOpData)>, Error> {
    // This is a method call to a variadic type.
    // Check if the method itself is variadic.
    let method2op = match vtd.name {
        "Array" => HashMap::from([]),
        "Shape" => HashMap::from([]),
        "PointerTile" => {
            HashMap::from([("broadcast", "broadcast_ptr"), ("reshape", "reshape_ptr")])
        }
        "Tensor" => HashMap::from([
            ("partition", "make_partition_view"),
            ("partition_permuted", "make_partition_view_permuted"),
            ("partition_mut", "make_partition_view_mut"),
            ("load_tile", "load_tile"),
            ("load", "load_tile_mut"),
            ("store", "store_tile"),
            ("shape", "get_tensor_shape"),
        ]),
        "Partition" => HashMap::from([("load", "load_from_view")]),
        "PartitionMut" => HashMap::from([
            ("load", "load_from_view_mut"),
            ("store", "store_to_view_mut"),
        ]),
        "Tile" => HashMap::from([("reshape", "reshape"), ("broadcast", "broadcast")]),
        "BroadcastScalar" => HashMap::from([("broadcast", "broadcast_scalar")]),
        _ => {
            return Span::call_site()
                .error(&format!("Unexpected variadic type: {}", vtd.name))
                .into()
        }
    };
    match method2op.get(method_name) {
        Some(op_name) => Ok(Some((
            op_name,
            get_variadic_op_data(op_name)
                .unwrap_or_else(|| panic!("{op_name} is not a variadic op.")),
        ))),
        None => Ok(None),
    }
}

/// Retrieves variadic operation data for a standalone function.
///
/// Standalone variadic functions (like `broadcast_scalar`, `reduce`, `scan`) have
/// their operation data looked up by function name rather than by type+method.
///
/// ## Parameters
///
/// - `op_name`: The function name (e.g., "broadcast_scalar", "reduce_min", "scan")
///
/// ## Returns
///
/// `Some(VariadicOpData)` if the function is variadic, `None` otherwise
///
/// ## Supported Variadic Operations
///
/// - **Type Conversions**: `broadcast_scalar`, `convert_tile`
/// - **Reductions**: `reduce`, `reduce_min`, `reduce_max`, `reduce_sum`, `reduce_prod`
/// - **Scans**: `scan`, `scan_sum`
/// - **Shape Operations**: `reshape`, `broadcast`, `permute`, `constant`
/// - **Extraction/Concatenation**: `extract`, `cat`
/// - **Element-wise Operations**: All arithmetic and comparison operations on tiles
/// - **Memory Operations**: `load_tile`, `load_tile_mut`, `store_tile`, `addptr`, `addptr_tile`
/// - **... and many more** (see implementation for full list)
///
/// ## Example
///
/// ```rust,ignore
/// let op_data = get_variadic_op_data("broadcast_scalar").unwrap();
/// // Describes type signature: <E, const S: [i32; N]>(E, Shape<S>) -> Tile<E, S>
/// ```
/// Returns variadic operation data for a standalone function, or `None` if not variadic.
pub fn get_variadic_op_data(op_name: &str) -> Option<VariadicOpData> {
    match op_name {
        "addptr" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("D", "N")]),
            input_map: vec![(0, "PointerTile", &["D"])],
            output_map: ("PointerTile", &["D"]),
            return_type: ("PointerTile", &["_", "D"]),
        }),
        "addptr_tile" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("D", "N")]),
            input_map: vec![(0, "PointerTile", &["D"]), (1, "Tile", &["D"])],
            output_map: ("PointerTile", &["D"]),
            return_type: ("PointerTile", &["_", "D"]),
        }),
        "make_tensor_view" => Some(VariadicOpData {
            const_length_vars: &["ZERO", "N"],
            cga_map: HashMap::from([("EMPTY", "ZERO"), ("D", "N"), ("C", "N")]),
            input_map: vec![
                (0, "PointerTile", &["EMPTY"]),
                (1, "Shape", &["D"]),
                (2, "Array", &["C"]),
            ],
            output_map: ("Tensor", &["D"]),
            return_type: ("Tensor", &["_", "D"]),
        }),
        "get_tensor_shape" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tensor", &["S"])],
            output_map: ("Shape", &["S"]),
            return_type: ("Shape", &["'_", "S"]),
        }),
        "get_shape_dim" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Shape", &["S"])],
            output_map: ("i32", &[]),
            return_type: ("i32", &[]),
        }),
        "get_tensor_token" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tensor", &["S"])],
            output_map: ("Token", &[]),
            return_type: ("Token", &[]),
        }),
        "set_tensor_token" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tensor", &["S"])],
            output_map: ("()", &[]),
            return_type: ("()", &[]),
        }),
        "make_partition_view" | "make_partition_view_padded" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("TENSOR_SHAPE", "N"), ("TILE_SHAPE", "N")]),
            input_map: vec![
                (0, "Tensor", &["TENSOR_SHAPE"]),
                (1, "Shape", &["TILE_SHAPE"]),
            ],
            output_map: ("Partition", &["TILE_SHAPE"]),
            return_type: ("Partition", &["'_", "_", "TILE_SHAPE"]),
        }),
        "make_partition_view_permuted" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("TENSOR_SHAPE", "N"), ("TILE_SHAPE", "N")]),
            input_map: vec![
                (0, "Tensor", &["TENSOR_SHAPE"]),
                (1, "Shape", &["TILE_SHAPE"]),
            ],
            output_map: ("Partition", &["TILE_SHAPE"]),
            return_type: ("Partition", &["'_", "_", "TILE_SHAPE"]),
        }),
        "get_partition_token" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("D", "N")]),
            input_map: vec![(0, "Partition", &["D"])],
            output_map: ("Token", &[]),
            return_type: ("Token", &[]),
        }),
        "load_from_view" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("D", "N")]),
            input_map: vec![(0, "Partition", ["D"].as_slice())],
            output_map: ("Tile", &["D"]),
            return_type: ("Tile", &["_", "D"]),
        }),
        "make_partition_view_mut" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("TENSOR_SHAPE", "N"), ("TILE_SHAPE", "N")]),
            input_map: vec![
                (0, "Tensor", &["TENSOR_SHAPE"]),
                (1, "Shape", &["TILE_SHAPE"]),
            ],
            output_map: ("PartitionMut", &["TILE_SHAPE"]),
            return_type: ("PartitionMut", &["'_", "_", "TILE_SHAPE"]),
        }),
        "get_partition_token_mut" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("D", "N")]),
            input_map: vec![(0, "PartitionMut", &["D"])],
            output_map: ("Token", &[]),
            return_type: ("Token", &[]),
        }),
        "set_partition_tensor_token" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("D", "N")]),
            input_map: vec![(0, "PartitionMut", &["D"])],
            output_map: ("()", &[]),
            return_type: ("()", &[]),
        }),
        "load_from_view_mut" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("D", "N")]),
            input_map: vec![(0, "PartitionMut", &["D"])],
            output_map: ("Tile", &["D"]),
            return_type: ("Tile", &["_", "D"]),
        }),
        "store_to_view_mut" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("D", "N")]),
            input_map: vec![(0, "PartitionMut", &["D"]), (1, "Tile", &["D"])],
            output_map: ("()", &[]),
            return_type: ("()", &[]),
        }),
        "reshape" => Some(VariadicOpData {
            const_length_vars: &["N", "M"],
            cga_map: HashMap::from([("S", "N"), ("R", "M")]),
            input_map: vec![(0, "Tile", &["S"]), (1, "Shape", &["R"])],
            output_map: ("Tile", &["R"]),
            return_type: ("Tile", &["_", "R"]),
        }),
        "broadcast_ptr" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N"), ("R", "N")]),
            input_map: vec![(0, "PointerTile", &["S"]), (1, "Shape", &["R"])],
            output_map: ("PointerTile", &["R"]),
            return_type: ("PointerTile", &["_", "R"]),
        }),
        "reshape_ptr" => Some(VariadicOpData {
            const_length_vars: &["N", "M"],
            cga_map: HashMap::from([("S", "N"), ("R", "M")]),
            input_map: vec![(0, "PointerTile", &["S"]), (1, "Shape", &["R"])],
            output_map: ("PointerTile", &["R"]),
            return_type: ("PointerTile", &["_", "R"]),
        }),
        "broadcast" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N"), ("R", "N")]),
            input_map: vec![(0, "Tile", &["S"]), (1, "Shape", &["R"])],
            output_map: ("Tile", &["R"]),
            return_type: ("Tile", &["_", "R"]),
        }),
        "permute" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("A", "N"), ("I", "N"), ("R", "N")]),
            input_map: vec![(0, "Tile", &["A"]), (1, "Array", &["I"])],
            output_map: ("Tile", &["R"]),
            return_type: ("Tile", &["_", "R"]),
        }),
        "constant" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(1, "Shape", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        "broadcast_scalar" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(1, "Shape", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        "load_tile" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N"), ("R", "N")]),
            input_map: vec![(0, "Tensor", &["S"]), (1, "Shape", &["R"])],
            output_map: ("Tile", &["R"]),
            return_type: ("Tile", &["_", "R"]),
        }),
        "load_tile_mut" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tensor", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        "store_tile" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tensor", &["S"]), (1, "Tile", &["S"])],
            output_map: ("()", &[]),
            return_type: ("()", &[]),
        }),
        "convert_tile" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tile", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        "check_partition_access" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Partition", &["S"])],
            output_map: ("()", &[]),
            return_type: ("()", &[]),
        }),
        "fma" | "fma_op" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![
                (0, "Tile", &["S"]),
                (1, "Tile", &["S"]),
                (2, "Tile", &["S"]),
            ],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        // Special element-wise binary ops.
        "min_tile" | "max_tile" | "true_div" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tile", &["S"]), (1, "Tile", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        "reduce_min" | "reduce_max" | "reduce_sum" | "reduce_prod" => Some(VariadicOpData {
            const_length_vars: &["N", "M"],
            cga_map: HashMap::from([("S", "N"), ("R", "M")]),
            input_map: vec![(0, "Tile", &["S"])],
            output_map: ("Tile", &["R"]),
            return_type: ("Tile", &["_", "R"]),
        }),
        // Unary operations.
        "ceil" | "cosh" | "cos" | "exp" | "exp2" | "exp2_ftz" | "log" | "log2" | "rsqrt"
        | "sinh" | "sin" | "sqrt" | "tanh" | "tan" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tile", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        // iot is not (yet?) variadic.
        // "iota" => Some(VariadicOpData {
        //     const_length_vars: &["N"],
        //     cga_map: HashMap::from([("S", "N")]),
        //     input_map: vec![(0, "Shape", &["S"])],
        //     output_map: ("Tile", &["S"]),
        //     return_type: ("Tile", &["_", "S"]),
        // }),
        "select" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![
                (0, "Tile", &["S"]),
                (1, "Tile", &["S"]),
                (2, "Tile", &["S"]),
            ],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        "eq_tile" | "ne_tile" | "gt_tile" | "ge_tile" | "lt_tile" | "le_tile" => {
            Some(VariadicOpData {
                const_length_vars: &["N"],
                cga_map: HashMap::from([("S", "N")]),
                input_map: vec![(0, "Tile", &["S"]), (1, "Tile", &["S"])],
                output_map: ("Tile", &["S"]),
                return_type: ("Tile", &["_", "S"]),
            })
        }
        // Additional unary operations not covered above
        "absf" | "absi" | "negf" | "negi" | "floor" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tile", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        "pow" | "maxf" | "maxf_ftz" | "minf" | "minf_ftz" | "andi" | "ori" | "xori" | "shli"
        | "shri" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tile", &["S"]), (1, "Tile", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        "bitcast" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tile", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        // cat is defined later with the same pattern
        "atomic_rmw_tko" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "PointerTile", &["S"]), (1, "Tile", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]), // Simplified - just track the Tile part
        }),
        "atomic_cas_tko" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![
                (0, "PointerTile", &["S"]),
                (1, "Tile", &["S"]),
                (2, "Tile", &["S"]),
            ],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]), // Simplified - just track the Tile part
        }),
        "load_ptr_tko" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "PointerTile", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        "store_ptr_tko" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "PointerTile", &["S"]), (1, "Tile", &["S"])],
            output_map: ("Token", &[]),
            return_type: ("Token", &[]),
        }),
        "maxi" | "mulhii" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tile", &["S"]), (1, "Tile", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        "exti" | "trunci" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tile", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        "scan_sum" | "reduce" | "scan" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tile", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        "int_to_ptr" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "Tile", &["S"])],
            output_map: ("PointerTile", &["S"]),
            return_type: ("PointerTile", &["_", "S"]),
        }),
        "ptr_to_int" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "PointerTile", &["S"])],
            output_map: ("Tile", &["S"]),
            return_type: ("Tile", &["_", "S"]),
        }),
        "ptr_to_ptr" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("S", "N")]),
            input_map: vec![(0, "PointerTile", &["S"])],
            output_map: ("PointerTile", &["S"]),
            return_type: ("PointerTile", &["_", "S"]),
        }),
        "extract" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("SIn", "N"), ("SOut", "N")]),
            input_map: vec![(0, "Tile", &["SIn"])],
            output_map: ("Tile", &["SOut"]),
            return_type: ("Tile", &["_", "SOut"]),
        }),
        "cat" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("SLhs", "N"), ("SRhs", "N"), ("SOut", "N")]),
            input_map: vec![(0, "Tile", &["SLhs"]), (1, "Tile", &["SRhs"])],
            output_map: ("Tile", &["SOut"]),
            return_type: ("Tile", &["_", "SOut"]),
        }),
        "load_tensor" => Some(VariadicOpData {
            const_length_vars: &["N", "M"],
            cga_map: HashMap::from([("S", "N"), ("R", "M")]),
            input_map: vec![(0, "Tensor", &["S"]), (2, "Shape", &["R"])],
            output_map: ("Tensor", &["R"]),
            return_type: ("Tensor", &["_", "R"]),
        }),
        "permute_array" => Some(VariadicOpData {
            const_length_vars: &["N"],
            cga_map: HashMap::from([("I", "N")]),
            input_map: vec![(1, "Array", &["I"])],
            output_map: ("()", &[]),
            return_type: ("[i32; N]", &[]),
        }),
        // "permute" => Some(VariadicOpData {
        //     const_length_vars: &["N"],
        //     cga_map: HashMap::from([("A", "N"), ("I", "N"), ("R", "N")]),
        //     input_map: vec![(0, "Tile", &["A"]), (1, "Array", &["I"])],
        //     output_map: ("Tile", &["R"]),
        //     return_type: ("Tile", &["_", "R"]),
        // }),
        // "load_tile_like" => Some(VariadicOpData {
        //     const_length_vars: &["N"],
        //     cga_map: HashMap::from([("S", "N")]),
        //     input_map: vec![(1, "Tensor", &["S"])],
        //     output_map: ("Tile", &["S"]),
        //     return_type: ("Tile", &["_", "S"]),
        // }),
        _ => None,
    }
}

/// Checks if a function name corresponds to a variadic operation.
///
/// ## Parameters
///
/// - `op_name`: The function name to check
///
/// ## Returns
///
/// `true` if the operation is variadic, `false` otherwise
///
/// ## Example
///
/// ```rust,ignore
/// assert!(is_variadic_op("broadcast_scalar"));
/// assert!(is_variadic_op("reduce_sum"));
/// assert!(!is_variadic_op("mma"));  // mma has fixed 2D/3D signatures
/// ```
/// Returns `true` if the given function name is a known variadic operation.
pub fn is_variadic_op(op_name: &str) -> bool {
    get_variadic_op_data(op_name).is_some()
}

// Represents an instance of a type with const generic arrays (one of VARIADIC_TYPES).
#[derive(Debug, Clone)]
/// Represents a specific instantiation of const-generic array parameters.
///
/// For variadic types with multiple CGAs, this struct holds the concrete values
/// for one specific instantiation. Used during macro expansion to generate
/// concrete types for all combinations of ranks.
///
/// ## Fields
///
/// - **`cga_arg_strings`**: String representations of CGA values (for codegen)
///   - Example: `[Some("S"), Some("R")]` for generic parameters
///   - Example: `[Some("[128, 64]"), None]` for concrete values
/// - **`n`**: Ranks for each CGA
///   - Example: `[2, 3]` means first CGA is 2D, second is 3D
///
/// ## Example
///
/// For `Tile<f32, {[128, 64]}>`:
/// ```rust,ignore
/// ConstGenericArrayType {
///     cga_arg_strings: vec![Some("S")],  // or Some("[128, 64]") for concrete
///     n: vec![2],                         // 2D
/// }
/// ```
/// A specific instantiation of const-generic array parameters with concrete ranks.
pub struct ConstGenericArrayType {
    pub cga_arg_strings: Vec<Option<String>>,
    pub n: Vec<u32>,
}

/// Generates a function name suffix for variadic operations.
///
/// Variadic operations are generated multiple times with rank-specific suffixes.
/// For example, `reshape` becomes `reshape_2_3` for reshaping from 2D to 3D.
///
/// ## Parameters
///
/// - `const_ga_lengths`: Vector of ranks for each CGA parameter
///
/// ## Returns
///
/// A string suffix like `"_2_3"` for operations with ranks [2, 3]
///
/// ## Example
///
/// ```rust,ignore
/// let suffix = get_variadic_function_suffix(&vec![2, 3]);
/// assert_eq!(suffix, "_2_3");
/// ```
/// Generates a function name suffix from CGA rank values (e.g., `[2, 3]` → `"2__3"`).
pub fn get_variadic_function_suffix(const_ga_lengths: &[u32]) -> String {
    const_ga_lengths
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>()
        .join("__")
}

/// Iterator over all combinations of ranks for a variadic type.
///
/// This iterator generates all possible instantiations of a variadic type
/// by enumerating through rank combinations. For example, if a type has
/// two CGA parameters that each can have ranks 0-4, this iterates through
/// all 25 combinations (5×5).
///
/// ## Fields
///
/// - **`i`**: Current iteration index
/// - **`i_max`**: Maximum iteration count (product of all rank ranges)
/// - **`n_vec`**: Maximum ranks for each CGA parameter
///
/// ## Example
///
/// ```rust,ignore
/// // For Tile with one CGA that goes from 0D to 4D:
/// let iter = ConstGenericArrayTypeIterator::new(&vec![5]);  // 5 ranks: 0,1,2,3,4
/// // Generates: Tile_0, Tile_1, Tile_2, Tile_3, Tile_4
/// ```
#[derive(Debug, Clone)]
/// Iterator over all rank combinations for a single variadic type's CGA parameters.
pub struct ConstGenericArrayTypeIterator {
    i: u32,
    i_max: u32,
    n_vec: Vec<u32>,
}

impl ConstGenericArrayTypeIterator {
    /// Creates a new iterator over rank combinations.
    ///
    /// ## Parameters
    ///
    /// - `n_vec`: Maximum ranks for each CGA parameter (e.g., `[5, 4]` means
    ///   first CGA can be 0-4D, second can be 0-3D)
    ///
    /// ## Returns
    ///
    /// An iterator that will generate all combinations
    pub fn new(n_vec: &[u32]) -> Self {
        Self {
            i: 0,
            n_vec: n_vec.to_vec(),
            i_max: n_vec.iter().product(),
        }
    }

    /// Creates a fresh iterator with the same configuration.
    ///
    /// Resets the iteration index to 0 while keeping the same rank configuration.
    pub fn renew(&self) -> ConstGenericArrayTypeIterator {
        Self {
            i: 0,
            n_vec: self.n_vec.clone(),
            i_max: self.i_max,
        }
    }
}

impl Iterator for ConstGenericArrayTypeIterator {
    type Item = ConstGenericArrayType;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.i_max {
            None
        } else {
            let mut result: Vec<u32> = vec![];
            let mut i = self.i;
            self.i += 1;
            for n in &self.n_vec {
                let r = i % n;
                i /= n;
                result.push(r);
            }
            let var_names = vec![None; self.n_vec.len()];
            Some(ConstGenericArrayType {
                cga_arg_strings: var_names,
                n: result,
            })
        }
    }
}

#[derive(Debug, Clone)]
/// Iterator that generates combinations across multiple type parameters.
///
/// While `ConstGenericArrayTypeIterator` handles a single type's rank combinations,
/// this iterator handles multiple types simultaneously. Used for operations that
/// involve multiple variadic types with different rank configurations.
///
/// ## Fields
///
/// - **`iterators`**: One iterator per type parameter
/// - **`state`**: Current instantiation for each type
/// - **`done`**: Whether iteration is complete
///
/// ## Example
///
/// For a function with two variadic parameters:
/// ```rust,ignore
/// // fn reshape<const S: [i32; N], const R: [i32; M]>(...)
/// let iter1 = ConstGenericArrayTypeIterator::new(&vec![5]);  // Input ranks 0-4
/// let iter2 = ConstGenericArrayTypeIterator::new(&vec![5]);  // Output ranks 0-4
/// let list_iter = ConstGenericArrayTypeListIterator::new(vec![iter1, iter2]);
/// // Generates all 25 combinations: (0,0), (0,1), ..., (4,4)
/// ```
/// Iterator over rank combinations across multiple variadic type parameters simultaneously.
pub struct ConstGenericArrayTypeListIterator {
    iterators: Vec<ConstGenericArrayTypeIterator>,
    state: Vec<ConstGenericArrayType>,
    done: bool,
}

impl ConstGenericArrayTypeListIterator {
    /// Creates a new multi-type iterator.
    ///
    /// ## Parameters
    ///
    /// - `iterators`: One iterator for each type parameter in the operation
    ///
    /// ## Returns
    ///
    /// An iterator that generates all combinations across all type parameters
    pub fn new(iterators: Vec<ConstGenericArrayTypeIterator>) -> Self {
        Self {
            iterators,
            state: vec![],
            done: false,
        }
    }
}

impl Iterator for ConstGenericArrayTypeListIterator {
    type Item = Result<Vec<ConstGenericArrayType>, Error>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.state.is_empty() {
            // First pass should always contain something.
            for item in &mut self.iterators {
                match item.next() {
                    Some(item) => {
                        self.state.push(item);
                    }
                    None => {
                        let error = Span::call_site().error(
                            "ConstGenericArrayTypeListIterator: iterator was empty on first pass.",
                        );
                        return Some(error.into());
                    }
                }
            }
            Some(Ok(self.state.clone()))
        } else if self.done {
            None
        } else {
            for _i in 0..self.iterators.len() {
                // Traverse in reverse to remain consistent with traversal order of individual ConstGenericArrayIterator.
                // The traversal is a mixed-radix counter.
                // We're done when the most significant position is None.
                let i = (self.iterators.len() - 1) - _i;
                let iter = &mut self.iterators[i];
                let item: Option<ConstGenericArrayType> = iter.next();
                match item {
                    Some(item) => {
                        self.state[i] = item;
                        break;
                    }
                    None => {
                        if i == 0 {
                            self.done = true;
                            return None;
                        }
                        self.iterators[i] = iter.renew();
                        self.state[i] = self.iterators[i].next().unwrap();
                    }
                }
            }
            Some(Ok(self.state.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::types::{ConstGenericArrayTypeIterator, ConstGenericArrayTypeListIterator};

    #[test]
    fn test_cg_arr_iter() -> () {
        for item in ConstGenericArrayTypeIterator::new(&vec![3]) {
            println!("{:?}", item);
        }
        for item in ConstGenericArrayTypeIterator::new(&vec![3, 3]) {
            println!("{:?}", item);
        }
    }

    #[test]
    fn test_multi_cg_arr_iter() -> () {
        println!("[{}]", 3);
        let a = ConstGenericArrayTypeIterator::new(&vec![3]);
        for item in ConstGenericArrayTypeListIterator::new(vec![a]) {
            let item = item.unwrap();
            println!("{:?}", item);
        }
        println!("[{}, {}]", 3, 3);
        let a = ConstGenericArrayTypeIterator::new(&vec![3]);
        let b = ConstGenericArrayTypeIterator::new(&vec![3]);
        for item in ConstGenericArrayTypeListIterator::new(vec![a, b]) {
            let item = item.unwrap();
            println!("{:?}", item);
        }
        println!("[{}, {}]", 3, 3);
        let a = ConstGenericArrayTypeIterator::new(&vec![2, 3]);
        let b = ConstGenericArrayTypeIterator::new(&vec![4, 5]);
        for item in ConstGenericArrayTypeListIterator::new(vec![a, b]) {
            let item = item.unwrap();
            println!("{:?}", item);
        }
    }
}
