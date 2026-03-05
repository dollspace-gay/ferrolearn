//! ONNX model export for ferrolearn fitted models.
//!
//! This module provides the [`ToOnnx`] trait and supporting types for
//! converting fitted ferrolearn models into ONNX (Open Neural Network Exchange)
//! format. The generated `.onnx` files follow the ONNX protobuf schema and can
//! be loaded by ONNX Runtime, `tract`, or other ONNX-compatible runtimes.
//!
//! # Supported Models
//!
//! - [`FittedLinearRegression`](ferrolearn_linear::FittedLinearRegression) — ONNX `LinearRegressor` (ML op)
//! - [`FittedLogisticRegression`](ferrolearn_linear::FittedLogisticRegression) — ONNX `LinearClassifier` (ML op)
//! - [`FittedRidge`](ferrolearn_linear::FittedRidge) — ONNX `LinearRegressor` (ML op)
//! - [`FittedDecisionTreeClassifier`](ferrolearn_tree::FittedDecisionTreeClassifier) — ONNX `TreeEnsembleClassifier` (ML op)
//! - [`FittedDecisionTreeRegressor`](ferrolearn_tree::FittedDecisionTreeRegressor) — ONNX `TreeEnsembleRegressor` (ML op)
//! - [`FittedRandomForestClassifier`](ferrolearn_tree::FittedRandomForestClassifier) — ONNX `TreeEnsembleClassifier` (ML op)
//! - [`FittedRandomForestRegressor`](ferrolearn_tree::FittedRandomForestRegressor) — ONNX `TreeEnsembleRegressor` (ML op)
//! - [`FittedGradientBoostingRegressor`](ferrolearn_tree::FittedGradientBoostingRegressor) — ONNX `TreeEnsembleRegressor` (ML op)
//! - [`FittedGradientBoostingClassifier`](ferrolearn_tree::FittedGradientBoostingClassifier) — ONNX `TreeEnsembleClassifier` (ML op)
//!
//! # Example
//!
//! ```no_run
//! use ferrolearn_io::onnx::{ToOnnx, save_onnx};
//! use ferrolearn_linear::{LinearRegression, FittedLinearRegression};
//! use ferrolearn_core::Fit;
//! use ndarray::{Array1, Array2, array};
//!
//! let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
//! let y = array![1.0, 2.0, 3.0, 4.0];
//! let model = LinearRegression::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//!
//! let onnx_model = fitted.to_onnx().unwrap();
//! save_onnx(&onnx_model, "/tmp/model.onnx").unwrap();
//! ```

use std::fs;
use std::path::Path;

use ferrolearn_core::error::FerroError;

// ---------------------------------------------------------------------------
// ONNX IR types (minimal subset for export)
// ---------------------------------------------------------------------------

/// A complete ONNX model, consisting of metadata and a computation graph.
#[derive(Debug, Clone)]
pub struct OnnxModel {
    /// ONNX IR version (typically 7 for opset 15+).
    pub ir_version: i64,
    /// Operator set imports.
    pub opset_import: Vec<OperatorSetId>,
    /// The computation graph.
    pub graph: OnnxGraph,
    /// Producer name (e.g. "ferrolearn").
    pub producer_name: String,
    /// Model documentation string.
    pub doc_string: String,
}

/// An operator set import identifying which opset version to use.
#[derive(Debug, Clone)]
pub struct OperatorSetId {
    /// The domain ("" for default ONNX, "ai.onnx.ml" for ML ops).
    pub domain: String,
    /// The operator set version.
    pub version: i64,
}

/// An ONNX computation graph containing nodes, inputs, outputs, and initializers.
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    /// Name of the graph.
    pub name: String,
    /// The computation nodes.
    pub nodes: Vec<OnnxNode>,
    /// Input tensor descriptions.
    pub inputs: Vec<ValueInfo>,
    /// Output tensor descriptions.
    pub outputs: Vec<ValueInfo>,
    /// Constant tensors (weights, biases).
    pub initializers: Vec<TensorProto>,
}

/// A single computation node in the graph.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// The operator type (e.g. "LinearRegressor", "TreeEnsembleClassifier").
    pub op_type: String,
    /// The operator domain ("" or "ai.onnx.ml").
    pub domain: String,
    /// Node name (optional, for debugging).
    pub name: String,
    /// Attributes for the operator.
    pub attributes: Vec<AttributeProto>,
}

/// A named attribute attached to a node.
#[derive(Debug, Clone)]
pub struct AttributeProto {
    /// Attribute name.
    pub name: String,
    /// Attribute value.
    pub value: AttributeValue,
}

/// The value of an attribute.
#[derive(Debug, Clone)]
pub enum AttributeValue {
    /// A single float value.
    Float(f32),
    /// A single integer value.
    Int(i64),
    /// A string value.
    String(String),
    /// A list of floats.
    Floats(Vec<f32>),
    /// A list of integers.
    Ints(Vec<i64>),
    /// A list of strings.
    Strings(Vec<String>),
}

/// Description of a tensor value (name + shape + element type).
#[derive(Debug, Clone)]
pub struct ValueInfo {
    /// Tensor name.
    pub name: String,
    /// Element type (1 = FLOAT, 11 = DOUBLE).
    pub elem_type: i32,
    /// Shape dimensions (use -1 for dynamic dimensions).
    pub shape: Vec<i64>,
}

/// A tensor constant (initializer).
#[derive(Debug, Clone)]
pub struct TensorProto {
    /// Tensor name.
    pub name: String,
    /// Element type (1 = FLOAT, 11 = DOUBLE).
    pub data_type: i32,
    /// Dimensions.
    pub dims: Vec<i64>,
    /// Float data (when data_type = 1).
    pub float_data: Vec<f32>,
    /// Double data (when data_type = 11).
    pub double_data: Vec<f64>,
}

/// ONNX element type constants.
const ONNX_FLOAT: i32 = 1;

// ---------------------------------------------------------------------------
// Protobuf wire format writer
// ---------------------------------------------------------------------------

/// Protobuf wire types.
const WIRE_VARINT: u32 = 0;
const WIRE_LEN: u32 = 2;
const WIRE_FIXED32: u32 = 5;

/// Encode a varint (unsigned LEB128).
fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

/// Encode a field tag (field number + wire type).
fn encode_tag(buf: &mut Vec<u8>, field_number: u32, wire_type: u32) {
    encode_varint(buf, ((field_number as u64) << 3) | (wire_type as u64));
}

/// Encode a varint field.
fn encode_varint_field(buf: &mut Vec<u8>, field_number: u32, value: u64) {
    if value != 0 {
        encode_tag(buf, field_number, WIRE_VARINT);
        encode_varint(buf, value);
    }
}

/// Encode a varint field even if zero.
fn encode_varint_field_always(buf: &mut Vec<u8>, field_number: u32, value: u64) {
    encode_tag(buf, field_number, WIRE_VARINT);
    encode_varint(buf, value);
}

/// Encode a length-delimited bytes field.
fn encode_bytes_field(buf: &mut Vec<u8>, field_number: u32, data: &[u8]) {
    if !data.is_empty() {
        encode_tag(buf, field_number, WIRE_LEN);
        encode_varint(buf, data.len() as u64);
        buf.extend_from_slice(data);
    }
}

/// Encode a string field.
fn encode_string_field(buf: &mut Vec<u8>, field_number: u32, s: &str) {
    if !s.is_empty() {
        encode_bytes_field(buf, field_number, s.as_bytes());
    }
}

/// Encode a submessage field.
fn encode_submessage_field(buf: &mut Vec<u8>, field_number: u32, data: &[u8]) {
    encode_tag(buf, field_number, WIRE_LEN);
    encode_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

/// Encode a float (fixed32) field.
fn encode_float_field(buf: &mut Vec<u8>, field_number: u32, value: f32) {
    encode_tag(buf, field_number, WIRE_FIXED32);
    buf.extend_from_slice(&value.to_le_bytes());
}

// ---------------------------------------------------------------------------
// ONNX protobuf serialization
// ---------------------------------------------------------------------------

/// Serialize an `OperatorSetId` to protobuf bytes.
fn serialize_opset_id(opset: &OperatorSetId) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_string_field(&mut buf, 1, &opset.domain); // domain
    encode_varint_field(&mut buf, 2, opset.version as u64); // version
    buf
}

/// Serialize a `TensorProto` to protobuf bytes.
fn serialize_tensor_proto(tensor: &TensorProto) -> Vec<u8> {
    let mut buf = Vec::new();
    // dims: field 1, repeated int64
    for &d in &tensor.dims {
        encode_varint_field_always(&mut buf, 1, d as u64);
    }
    // data_type: field 2
    encode_varint_field(&mut buf, 2, tensor.data_type as u64);
    // float_data: field 4, packed repeated float
    if !tensor.float_data.is_empty() {
        let mut packed = Vec::with_capacity(tensor.float_data.len() * 4);
        for &f in &tensor.float_data {
            packed.extend_from_slice(&f.to_le_bytes());
        }
        encode_bytes_field(&mut buf, 4, &packed);
    }
    // double_data: field 5, packed repeated double
    if !tensor.double_data.is_empty() {
        let mut packed = Vec::with_capacity(tensor.double_data.len() * 8);
        for &d in &tensor.double_data {
            packed.extend_from_slice(&d.to_le_bytes());
        }
        encode_bytes_field(&mut buf, 5, &packed);
    }
    // name: field 8
    encode_string_field(&mut buf, 8, &tensor.name);
    buf
}

/// ONNX attribute type constants (from onnx.proto3 `AttributeProto.AttributeType`).
const ATTR_FLOAT: i32 = 1;
const ATTR_INT: i32 = 2;
const ATTR_STRING: i32 = 3;
const ATTR_FLOATS: i32 = 6;
const ATTR_INTS: i32 = 7;
const ATTR_STRINGS: i32 = 8;

/// Serialize an `AttributeProto` to protobuf bytes.
fn serialize_attribute(attr: &AttributeProto) -> Vec<u8> {
    let mut buf = Vec::new();
    // name: field 1
    encode_string_field(&mut buf, 1, &attr.name);

    match &attr.value {
        AttributeValue::Float(v) => {
            encode_float_field(&mut buf, 2, *v); // f: field 2
            encode_varint_field(&mut buf, 20, ATTR_FLOAT as u64); // type: field 20
        }
        AttributeValue::Int(v) => {
            encode_tag(&mut buf, 3, WIRE_VARINT); // i: field 3
            encode_varint(&mut buf, *v as u64);
            encode_varint_field(&mut buf, 20, ATTR_INT as u64);
        }
        AttributeValue::String(v) => {
            encode_bytes_field(&mut buf, 4, v.as_bytes()); // s: field 4
            encode_varint_field(&mut buf, 20, ATTR_STRING as u64);
        }
        AttributeValue::Floats(vs) => {
            // floats: field 7, packed repeated float
            let mut packed = Vec::with_capacity(vs.len() * 4);
            for &f in vs {
                packed.extend_from_slice(&f.to_le_bytes());
            }
            encode_bytes_field(&mut buf, 7, &packed);
            encode_varint_field(&mut buf, 20, ATTR_FLOATS as u64);
        }
        AttributeValue::Ints(vs) => {
            // ints: field 8, packed repeated int64
            let mut packed = Vec::new();
            for &i in vs {
                encode_varint(&mut packed, i as u64);
            }
            encode_bytes_field(&mut buf, 8, &packed);
            encode_varint_field(&mut buf, 20, ATTR_INTS as u64);
        }
        AttributeValue::Strings(vs) => {
            // strings: field 9, repeated bytes (not packed)
            for s in vs {
                encode_bytes_field(&mut buf, 9, s.as_bytes());
            }
            encode_varint_field(&mut buf, 20, ATTR_STRINGS as u64);
        }
    }

    buf
}

/// Serialize an `OnnxNode` (NodeProto) to protobuf bytes.
fn serialize_node(node: &OnnxNode) -> Vec<u8> {
    let mut buf = Vec::new();
    // input: field 1, repeated string
    for input in &node.inputs {
        encode_string_field(&mut buf, 1, input);
    }
    // output: field 2, repeated string
    for output in &node.outputs {
        encode_string_field(&mut buf, 2, output);
    }
    // name: field 3
    encode_string_field(&mut buf, 3, &node.name);
    // op_type: field 4
    encode_string_field(&mut buf, 4, &node.op_type);
    // domain: field 7
    encode_string_field(&mut buf, 7, &node.domain);
    // attribute: field 5, repeated
    for attr in &node.attributes {
        let attr_bytes = serialize_attribute(attr);
        encode_submessage_field(&mut buf, 5, &attr_bytes);
    }
    buf
}

/// Serialize a `TypeProto` (for ValueInfo) to protobuf bytes.
fn serialize_type_proto(elem_type: i32, shape: &[i64]) -> Vec<u8> {
    // TensorTypeProto
    let mut tensor_buf = Vec::new();
    // elem_type: field 1
    encode_varint_field(&mut tensor_buf, 1, elem_type as u64);
    // shape: field 2 (TensorShapeProto)
    let mut shape_buf = Vec::new();
    for &dim in shape {
        // dim: repeated Dimension (field 1)
        let mut dim_buf = Vec::new();
        if dim >= 0 {
            // dim_value: field 1 in Dimension
            encode_varint_field_always(&mut dim_buf, 1, dim as u64);
        } else {
            // dim_param: field 2 in Dimension (dynamic dim)
            encode_string_field(&mut dim_buf, 2, "N");
        }
        encode_submessage_field(&mut shape_buf, 1, &dim_buf);
    }
    encode_submessage_field(&mut tensor_buf, 2, &shape_buf);

    // TypeProto wraps tensor_type at field 1
    let mut type_buf = Vec::new();
    encode_submessage_field(&mut type_buf, 1, &tensor_buf);
    type_buf
}

/// Serialize a `ValueInfo` to protobuf bytes.
fn serialize_value_info(vi: &ValueInfo) -> Vec<u8> {
    let mut buf = Vec::new();
    // name: field 1
    encode_string_field(&mut buf, 1, &vi.name);
    // type: field 2
    let type_bytes = serialize_type_proto(vi.elem_type, &vi.shape);
    encode_submessage_field(&mut buf, 2, &type_bytes);
    buf
}

/// Serialize an `OnnxGraph` (GraphProto) to protobuf bytes.
fn serialize_graph(graph: &OnnxGraph) -> Vec<u8> {
    let mut buf = Vec::new();
    // node: field 1, repeated
    for node in &graph.nodes {
        let node_bytes = serialize_node(node);
        encode_submessage_field(&mut buf, 1, &node_bytes);
    }
    // name: field 2
    encode_string_field(&mut buf, 2, &graph.name);
    // initializer: field 5, repeated
    for init in &graph.initializers {
        let init_bytes = serialize_tensor_proto(init);
        encode_submessage_field(&mut buf, 5, &init_bytes);
    }
    // input: field 11, repeated
    for input in &graph.inputs {
        let vi_bytes = serialize_value_info(input);
        encode_submessage_field(&mut buf, 11, &vi_bytes);
    }
    // output: field 12, repeated
    for output in &graph.outputs {
        let vi_bytes = serialize_value_info(output);
        encode_submessage_field(&mut buf, 12, &vi_bytes);
    }
    buf
}

/// Serialize an `OnnxModel` (ModelProto) to protobuf bytes.
fn serialize_model(model: &OnnxModel) -> Vec<u8> {
    let mut buf = Vec::new();
    // ir_version: field 1
    encode_varint_field(&mut buf, 1, model.ir_version as u64);
    // opset_import: field 8, repeated
    for opset in &model.opset_import {
        let opset_bytes = serialize_opset_id(opset);
        encode_submessage_field(&mut buf, 8, &opset_bytes);
    }
    // producer_name: field 2
    encode_string_field(&mut buf, 2, &model.producer_name);
    // doc_string: field 6
    encode_string_field(&mut buf, 6, &model.doc_string);
    // graph: field 7
    let graph_bytes = serialize_graph(&model.graph);
    encode_submessage_field(&mut buf, 7, &graph_bytes);
    buf
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Trait for converting a fitted model into an ONNX model representation.
pub trait ToOnnx {
    /// Convert this fitted model to an [`OnnxModel`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::SerdeError`] if the conversion fails.
    fn to_onnx(&self) -> Result<OnnxModel, FerroError>;
}

/// Serialize an [`OnnxModel`] to bytes in ONNX protobuf format.
///
/// The returned bytes form a valid `.onnx` file that can be loaded by
/// ONNX Runtime, `tract`, or other ONNX-compatible runtimes.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if serialization fails.
pub fn save_onnx_bytes(model: &OnnxModel) -> Result<Vec<u8>, FerroError> {
    Ok(serialize_model(model))
}

/// Serialize an [`OnnxModel`] and write it to a file.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if serialization fails, or
/// [`FerroError::IoError`] if the file cannot be written.
pub fn save_onnx(model: &OnnxModel, path: impl AsRef<Path>) -> Result<(), FerroError> {
    let bytes = save_onnx_bytes(model)?;
    fs::write(path, bytes).map_err(FerroError::IoError)
}

// ---------------------------------------------------------------------------
// Helper: create a standard ONNX model skeleton
// ---------------------------------------------------------------------------

/// Create an ONNX model with standard metadata and the given graph.
fn make_onnx_model(graph: OnnxGraph) -> OnnxModel {
    OnnxModel {
        ir_version: 7,
        opset_import: vec![
            OperatorSetId {
                domain: String::new(),
                version: 15,
            },
            OperatorSetId {
                domain: "ai.onnx.ml".to_owned(),
                version: 2,
            },
        ],
        graph,
        producer_name: "ferrolearn".to_owned(),
        doc_string: String::new(),
    }
}

// ---------------------------------------------------------------------------
// ToOnnx implementations for linear models
// ---------------------------------------------------------------------------

use ferrolearn_core::introspection::{HasClasses, HasCoefficients};
use ndarray::Array1;
use num_traits::Float;

/// Helper to convert a float to f32.
fn to_f32<F: Float>(x: F) -> f32 {
    x.to_f64().unwrap_or(0.0) as f32
}

impl ToOnnx for ferrolearn_linear::FittedLinearRegression<f64> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        linear_regressor_to_onnx(
            self.coefficients(),
            self.intercept(),
            "FittedLinearRegression",
        )
    }
}

impl ToOnnx for ferrolearn_linear::FittedLinearRegression<f32> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        linear_regressor_to_onnx(
            self.coefficients(),
            self.intercept(),
            "FittedLinearRegression",
        )
    }
}

impl ToOnnx for ferrolearn_linear::FittedRidge<f64> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        linear_regressor_to_onnx(self.coefficients(), self.intercept(), "FittedRidge")
    }
}

impl ToOnnx for ferrolearn_linear::FittedRidge<f32> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        linear_regressor_to_onnx(self.coefficients(), self.intercept(), "FittedRidge")
    }
}

/// Build an ONNX model with a `LinearRegressor` ML op node.
fn linear_regressor_to_onnx<F: Float>(
    coefficients: &Array1<F>,
    intercept: F,
    model_name: &str,
) -> Result<OnnxModel, FerroError> {
    let n_features = coefficients.len();
    let coef_f32: Vec<f32> = coefficients.iter().map(|&c| to_f32(c)).collect();
    let intercept_f32 = to_f32(intercept);

    let node = OnnxNode {
        inputs: vec!["X".to_owned()],
        outputs: vec!["Y".to_owned()],
        op_type: "LinearRegressor".to_owned(),
        domain: "ai.onnx.ml".to_owned(),
        name: "linear_regressor".to_owned(),
        attributes: vec![
            AttributeProto {
                name: "coefficients".to_owned(),
                value: AttributeValue::Floats(coef_f32),
            },
            AttributeProto {
                name: "intercepts".to_owned(),
                value: AttributeValue::Floats(vec![intercept_f32]),
            },
            AttributeProto {
                name: "targets".to_owned(),
                value: AttributeValue::Int(1),
            },
        ],
    };

    let graph = OnnxGraph {
        name: model_name.to_owned(),
        nodes: vec![node],
        inputs: vec![ValueInfo {
            name: "X".to_owned(),
            elem_type: ONNX_FLOAT,
            shape: vec![-1, n_features as i64],
        }],
        outputs: vec![ValueInfo {
            name: "Y".to_owned(),
            elem_type: ONNX_FLOAT,
            shape: vec![-1, 1],
        }],
        initializers: vec![],
    };

    Ok(make_onnx_model(graph))
}

// ---------------------------------------------------------------------------
// ToOnnx for logistic regression
// ---------------------------------------------------------------------------

impl ToOnnx for ferrolearn_linear::FittedLogisticRegression<f64> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        logistic_regression_to_onnx(self)
    }
}

/// Build an ONNX model with a `LinearClassifier` ML op node.
fn logistic_regression_to_onnx(
    model: &ferrolearn_linear::FittedLogisticRegression<f64>,
) -> Result<OnnxModel, FerroError> {
    let weight_matrix = model.weight_matrix();
    let intercept_vec = model.intercept_vec();
    let classes = model.classes();
    let n_features = weight_matrix.ncols();

    // Flatten weight matrix to coefficients
    let coefficients: Vec<f32> = weight_matrix.iter().map(|&v| v as f32).collect();
    let intercepts: Vec<f32> = intercept_vec.iter().map(|&v| v as f32).collect();
    let class_labels: Vec<i64> = classes.iter().map(|&c| c as i64).collect();

    let post_transform = if model.is_binary() {
        "LOGISTIC"
    } else {
        "SOFTMAX"
    };

    let node = OnnxNode {
        inputs: vec!["X".to_owned()],
        outputs: vec!["label".to_owned(), "probabilities".to_owned()],
        op_type: "LinearClassifier".to_owned(),
        domain: "ai.onnx.ml".to_owned(),
        name: "linear_classifier".to_owned(),
        attributes: vec![
            AttributeProto {
                name: "coefficients".to_owned(),
                value: AttributeValue::Floats(coefficients),
            },
            AttributeProto {
                name: "intercepts".to_owned(),
                value: AttributeValue::Floats(intercepts),
            },
            AttributeProto {
                name: "classlabels_ints".to_owned(),
                value: AttributeValue::Ints(class_labels),
            },
            AttributeProto {
                name: "post_transform".to_owned(),
                value: AttributeValue::String(post_transform.to_owned()),
            },
        ],
    };

    let graph = OnnxGraph {
        name: "FittedLogisticRegression".to_owned(),
        nodes: vec![node],
        inputs: vec![ValueInfo {
            name: "X".to_owned(),
            elem_type: ONNX_FLOAT,
            shape: vec![-1, n_features as i64],
        }],
        outputs: vec![
            ValueInfo {
                name: "label".to_owned(),
                elem_type: 7, // INT64
                shape: vec![-1],
            },
            ValueInfo {
                name: "probabilities".to_owned(),
                elem_type: ONNX_FLOAT,
                shape: vec![-1, classes.len() as i64],
            },
        ],
        initializers: vec![],
    };

    Ok(make_onnx_model(graph))
}

// ---------------------------------------------------------------------------
// ToOnnx for decision trees and tree ensembles
// ---------------------------------------------------------------------------

use ferrolearn_tree::Node;

/// Flatten a tree for classifier ONNX export, with class distribution weights.
#[allow(clippy::type_complexity)]
fn flatten_tree_classifier_for_onnx<F: Float>(
    nodes: &[Node<F>],
    tree_id: i64,
    n_classes: usize,
) -> (
    Vec<i64>,    // nodes_treeids
    Vec<i64>,    // nodes_nodeids
    Vec<i64>,    // nodes_featureids
    Vec<f32>,    // nodes_values
    Vec<String>, // nodes_modes
    Vec<i64>,    // nodes_truenodeids
    Vec<i64>,    // nodes_falsenodeids
    Vec<i64>,    // class_treeids
    Vec<i64>,    // class_nodeids
    Vec<i64>,    // class_ids
    Vec<f32>,    // class_weights
) {
    let mut treeids = Vec::new();
    let mut nodeids = Vec::new();
    let mut featureids = Vec::new();
    let mut values = Vec::new();
    let mut modes = Vec::new();
    let mut truenodeids = Vec::new();
    let mut falsenodeids = Vec::new();
    let mut class_treeids = Vec::new();
    let mut class_nodeids = Vec::new();
    let mut class_ids = Vec::new();
    let mut class_weights = Vec::new();

    for (idx, node) in nodes.iter().enumerate() {
        treeids.push(tree_id);
        nodeids.push(idx as i64);

        match node {
            Node::Split {
                feature,
                threshold,
                left,
                right,
                ..
            } => {
                featureids.push(*feature as i64);
                values.push(to_f32(*threshold));
                modes.push("BRANCH_LEQ".to_owned());
                truenodeids.push(*left as i64);
                falsenodeids.push(*right as i64);
            }
            Node::Leaf {
                class_distribution, ..
            } => {
                featureids.push(0);
                values.push(0.0);
                modes.push("LEAF".to_owned());
                truenodeids.push(0);
                falsenodeids.push(0);

                if let Some(dist) = class_distribution {
                    for (c, &w) in dist.iter().enumerate() {
                        class_treeids.push(tree_id);
                        class_nodeids.push(idx as i64);
                        class_ids.push(c as i64);
                        class_weights.push(to_f32(w));
                    }
                } else {
                    // Fallback: single class
                    for c in 0..n_classes {
                        class_treeids.push(tree_id);
                        class_nodeids.push(idx as i64);
                        class_ids.push(c as i64);
                        class_weights.push(0.0);
                    }
                }
            }
        }
    }

    (
        treeids,
        nodeids,
        featureids,
        values,
        modes,
        truenodeids,
        falsenodeids,
        class_treeids,
        class_nodeids,
        class_ids,
        class_weights,
    )
}

// --- DecisionTreeRegressor ---

impl ToOnnx for ferrolearn_tree::FittedDecisionTreeRegressor<f64> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        tree_regressor_to_onnx(&[self.nodes()], self.n_features(), "SUM")
    }
}

impl ToOnnx for ferrolearn_tree::FittedDecisionTreeRegressor<f32> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        tree_regressor_to_onnx(&[self.nodes()], self.n_features(), "SUM")
    }
}

/// Build an ONNX model with a `TreeEnsembleRegressor` ML op node from one or more trees.
fn tree_regressor_to_onnx<F: Float>(
    trees: &[&[Node<F>]],
    n_features: usize,
    aggregate_function: &str,
) -> Result<OnnxModel, FerroError> {
    let mut all_treeids = Vec::new();
    let mut all_nodeids = Vec::new();
    let mut all_featureids = Vec::new();
    let mut all_values = Vec::new();
    let mut all_modes = Vec::new();
    let mut all_truenodeids = Vec::new();
    let mut all_falsenodeids = Vec::new();
    let mut all_target_treeids = Vec::new();
    let mut all_target_ids = Vec::new();
    let mut all_target_weights = Vec::new();
    let mut all_target_nodeids = Vec::new();

    for (tid, tree) in trees.iter().enumerate() {
        for (idx, node) in tree.iter().enumerate() {
            all_treeids.push(tid as i64);
            all_nodeids.push(idx as i64);

            match node {
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    ..
                } => {
                    all_featureids.push(*feature as i64);
                    all_values.push(to_f32(*threshold));
                    all_modes.push("BRANCH_LEQ".to_owned());
                    all_truenodeids.push(*left as i64);
                    all_falsenodeids.push(*right as i64);
                }
                Node::Leaf { value, .. } => {
                    all_featureids.push(0);
                    all_values.push(0.0);
                    all_modes.push("LEAF".to_owned());
                    all_truenodeids.push(0);
                    all_falsenodeids.push(0);

                    all_target_treeids.push(tid as i64);
                    all_target_nodeids.push(idx as i64);
                    all_target_ids.push(0_i64);
                    all_target_weights.push(to_f32(*value));
                }
            }
        }
    }

    let node = OnnxNode {
        inputs: vec!["X".to_owned()],
        outputs: vec!["Y".to_owned()],
        op_type: "TreeEnsembleRegressor".to_owned(),
        domain: "ai.onnx.ml".to_owned(),
        name: "tree_ensemble_regressor".to_owned(),
        attributes: vec![
            AttributeProto {
                name: "nodes_treeids".to_owned(),
                value: AttributeValue::Ints(all_treeids),
            },
            AttributeProto {
                name: "nodes_nodeids".to_owned(),
                value: AttributeValue::Ints(all_nodeids),
            },
            AttributeProto {
                name: "nodes_featureids".to_owned(),
                value: AttributeValue::Ints(all_featureids),
            },
            AttributeProto {
                name: "nodes_values".to_owned(),
                value: AttributeValue::Floats(all_values),
            },
            AttributeProto {
                name: "nodes_modes".to_owned(),
                value: AttributeValue::Strings(all_modes),
            },
            AttributeProto {
                name: "nodes_truenodeids".to_owned(),
                value: AttributeValue::Ints(all_truenodeids),
            },
            AttributeProto {
                name: "nodes_falsenodeids".to_owned(),
                value: AttributeValue::Ints(all_falsenodeids),
            },
            AttributeProto {
                name: "target_treeids".to_owned(),
                value: AttributeValue::Ints(all_target_treeids),
            },
            AttributeProto {
                name: "target_nodeids".to_owned(),
                value: AttributeValue::Ints(all_target_nodeids),
            },
            AttributeProto {
                name: "target_ids".to_owned(),
                value: AttributeValue::Ints(all_target_ids),
            },
            AttributeProto {
                name: "target_weights".to_owned(),
                value: AttributeValue::Floats(all_target_weights),
            },
            AttributeProto {
                name: "n_targets".to_owned(),
                value: AttributeValue::Int(1),
            },
            AttributeProto {
                name: "aggregate_function".to_owned(),
                value: AttributeValue::String(aggregate_function.to_owned()),
            },
            AttributeProto {
                name: "post_transform".to_owned(),
                value: AttributeValue::String("NONE".to_owned()),
            },
        ],
    };

    let graph = OnnxGraph {
        name: "TreeEnsembleRegressor".to_owned(),
        nodes: vec![node],
        inputs: vec![ValueInfo {
            name: "X".to_owned(),
            elem_type: ONNX_FLOAT,
            shape: vec![-1, n_features as i64],
        }],
        outputs: vec![ValueInfo {
            name: "Y".to_owned(),
            elem_type: ONNX_FLOAT,
            shape: vec![-1, 1],
        }],
        initializers: vec![],
    };

    Ok(make_onnx_model(graph))
}

// --- DecisionTreeClassifier ---

impl ToOnnx for ferrolearn_tree::FittedDecisionTreeClassifier<f64> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        tree_classifier_to_onnx(&[self.nodes()], self.n_features(), self.classes())
    }
}

impl ToOnnx for ferrolearn_tree::FittedDecisionTreeClassifier<f32> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        tree_classifier_to_onnx(&[self.nodes()], self.n_features(), self.classes())
    }
}

/// Build an ONNX model with a `TreeEnsembleClassifier` ML op node.
fn tree_classifier_to_onnx<F: Float>(
    trees: &[&[Node<F>]],
    n_features: usize,
    classes: &[usize],
) -> Result<OnnxModel, FerroError> {
    let n_classes = classes.len();

    let mut all_treeids = Vec::new();
    let mut all_nodeids = Vec::new();
    let mut all_featureids = Vec::new();
    let mut all_values = Vec::new();
    let mut all_modes = Vec::new();
    let mut all_truenodeids = Vec::new();
    let mut all_falsenodeids = Vec::new();
    let mut all_class_treeids = Vec::new();
    let mut all_class_nodeids = Vec::new();
    let mut all_class_ids = Vec::new();
    let mut all_class_weights = Vec::new();

    for (tid, tree) in trees.iter().enumerate() {
        let (tids, nids, fids, vals, ms, tnids, fnids, ctids, cnids, cids, cwts) =
            flatten_tree_classifier_for_onnx(tree, tid as i64, n_classes);
        all_treeids.extend(tids);
        all_nodeids.extend(nids);
        all_featureids.extend(fids);
        all_values.extend(vals);
        all_modes.extend(ms);
        all_truenodeids.extend(tnids);
        all_falsenodeids.extend(fnids);
        all_class_treeids.extend(ctids);
        all_class_nodeids.extend(cnids);
        all_class_ids.extend(cids);
        all_class_weights.extend(cwts);
    }

    let class_labels: Vec<i64> = classes.iter().map(|&c| c as i64).collect();

    let node = OnnxNode {
        inputs: vec!["X".to_owned()],
        outputs: vec!["label".to_owned(), "probabilities".to_owned()],
        op_type: "TreeEnsembleClassifier".to_owned(),
        domain: "ai.onnx.ml".to_owned(),
        name: "tree_ensemble_classifier".to_owned(),
        attributes: vec![
            AttributeProto {
                name: "nodes_treeids".to_owned(),
                value: AttributeValue::Ints(all_treeids),
            },
            AttributeProto {
                name: "nodes_nodeids".to_owned(),
                value: AttributeValue::Ints(all_nodeids),
            },
            AttributeProto {
                name: "nodes_featureids".to_owned(),
                value: AttributeValue::Ints(all_featureids),
            },
            AttributeProto {
                name: "nodes_values".to_owned(),
                value: AttributeValue::Floats(all_values),
            },
            AttributeProto {
                name: "nodes_modes".to_owned(),
                value: AttributeValue::Strings(all_modes),
            },
            AttributeProto {
                name: "nodes_truenodeids".to_owned(),
                value: AttributeValue::Ints(all_truenodeids),
            },
            AttributeProto {
                name: "nodes_falsenodeids".to_owned(),
                value: AttributeValue::Ints(all_falsenodeids),
            },
            AttributeProto {
                name: "class_treeids".to_owned(),
                value: AttributeValue::Ints(all_class_treeids),
            },
            AttributeProto {
                name: "class_nodeids".to_owned(),
                value: AttributeValue::Ints(all_class_nodeids),
            },
            AttributeProto {
                name: "class_ids".to_owned(),
                value: AttributeValue::Ints(all_class_ids),
            },
            AttributeProto {
                name: "class_weights".to_owned(),
                value: AttributeValue::Floats(all_class_weights),
            },
            AttributeProto {
                name: "classlabels_int64s".to_owned(),
                value: AttributeValue::Ints(class_labels),
            },
            AttributeProto {
                name: "post_transform".to_owned(),
                value: AttributeValue::String("NONE".to_owned()),
            },
        ],
    };

    let graph = OnnxGraph {
        name: "TreeEnsembleClassifier".to_owned(),
        nodes: vec![node],
        inputs: vec![ValueInfo {
            name: "X".to_owned(),
            elem_type: ONNX_FLOAT,
            shape: vec![-1, n_features as i64],
        }],
        outputs: vec![
            ValueInfo {
                name: "label".to_owned(),
                elem_type: 7, // INT64
                shape: vec![-1],
            },
            ValueInfo {
                name: "probabilities".to_owned(),
                elem_type: ONNX_FLOAT,
                shape: vec![-1, n_classes as i64],
            },
        ],
        initializers: vec![],
    };

    Ok(make_onnx_model(graph))
}

// --- RandomForestClassifier ---

impl ToOnnx for ferrolearn_tree::FittedRandomForestClassifier<f64> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        let tree_refs: Vec<&[Node<f64>]> = self.trees().iter().map(|t| t.as_slice()).collect();
        tree_classifier_to_onnx(&tree_refs, self.n_features(), self.classes())
    }
}

impl ToOnnx for ferrolearn_tree::FittedRandomForestClassifier<f32> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        let tree_refs: Vec<&[Node<f32>]> = self.trees().iter().map(|t| t.as_slice()).collect();
        tree_classifier_to_onnx(&tree_refs, self.n_features(), self.classes())
    }
}

// --- RandomForestRegressor ---

impl ToOnnx for ferrolearn_tree::FittedRandomForestRegressor<f64> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        let tree_refs: Vec<&[Node<f64>]> = self.trees().iter().map(|t| t.as_slice()).collect();
        tree_regressor_to_onnx(&tree_refs, self.n_features(), "AVERAGE")
    }
}

impl ToOnnx for ferrolearn_tree::FittedRandomForestRegressor<f32> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        let tree_refs: Vec<&[Node<f32>]> = self.trees().iter().map(|t| t.as_slice()).collect();
        tree_regressor_to_onnx(&tree_refs, self.n_features(), "AVERAGE")
    }
}

// --- GradientBoostingRegressor ---

impl ToOnnx for ferrolearn_tree::FittedGradientBoostingRegressor<f64> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        gradient_boosting_regressor_to_onnx(self)
    }
}

/// Build an ONNX model for a gradient boosting regressor.
///
/// Uses `TreeEnsembleRegressor` with `SUM` aggregation. The initial prediction
/// and learning rate are baked into the leaf weights.
fn gradient_boosting_regressor_to_onnx(
    model: &ferrolearn_tree::FittedGradientBoostingRegressor<f64>,
) -> Result<OnnxModel, FerroError> {
    let n_features = model.n_features();
    let init = model.init() as f32;
    let lr = model.learning_rate() as f32;

    let mut all_treeids = Vec::new();
    let mut all_nodeids = Vec::new();
    let mut all_featureids = Vec::new();
    let mut all_values = Vec::new();
    let mut all_modes = Vec::new();
    let mut all_truenodeids = Vec::new();
    let mut all_falsenodeids = Vec::new();
    let mut all_target_treeids = Vec::new();
    let mut all_target_nodeids = Vec::new();
    let mut all_target_ids = Vec::new();
    let mut all_target_weights = Vec::new();

    for (tid, tree) in model.trees().iter().enumerate() {
        for (idx, node) in tree.iter().enumerate() {
            all_treeids.push(tid as i64);
            all_nodeids.push(idx as i64);

            match node {
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    ..
                } => {
                    all_featureids.push(*feature as i64);
                    all_values.push(*threshold as f32);
                    all_modes.push("BRANCH_LEQ".to_owned());
                    all_truenodeids.push(*left as i64);
                    all_falsenodeids.push(*right as i64);
                }
                Node::Leaf { value, .. } => {
                    all_featureids.push(0);
                    all_values.push(0.0);
                    all_modes.push("LEAF".to_owned());
                    all_truenodeids.push(0);
                    all_falsenodeids.push(0);

                    all_target_treeids.push(tid as i64);
                    all_target_nodeids.push(idx as i64);
                    all_target_ids.push(0_i64);
                    // Bake learning rate into leaf weights
                    all_target_weights.push(*value as f32 * lr);
                }
            }
        }
    }

    let node = OnnxNode {
        inputs: vec!["X".to_owned()],
        outputs: vec!["Y".to_owned()],
        op_type: "TreeEnsembleRegressor".to_owned(),
        domain: "ai.onnx.ml".to_owned(),
        name: "gradient_boosting_regressor".to_owned(),
        attributes: vec![
            AttributeProto {
                name: "nodes_treeids".to_owned(),
                value: AttributeValue::Ints(all_treeids),
            },
            AttributeProto {
                name: "nodes_nodeids".to_owned(),
                value: AttributeValue::Ints(all_nodeids),
            },
            AttributeProto {
                name: "nodes_featureids".to_owned(),
                value: AttributeValue::Ints(all_featureids),
            },
            AttributeProto {
                name: "nodes_values".to_owned(),
                value: AttributeValue::Floats(all_values),
            },
            AttributeProto {
                name: "nodes_modes".to_owned(),
                value: AttributeValue::Strings(all_modes),
            },
            AttributeProto {
                name: "nodes_truenodeids".to_owned(),
                value: AttributeValue::Ints(all_truenodeids),
            },
            AttributeProto {
                name: "nodes_falsenodeids".to_owned(),
                value: AttributeValue::Ints(all_falsenodeids),
            },
            AttributeProto {
                name: "target_treeids".to_owned(),
                value: AttributeValue::Ints(all_target_treeids),
            },
            AttributeProto {
                name: "target_nodeids".to_owned(),
                value: AttributeValue::Ints(all_target_nodeids),
            },
            AttributeProto {
                name: "target_ids".to_owned(),
                value: AttributeValue::Ints(all_target_ids),
            },
            AttributeProto {
                name: "target_weights".to_owned(),
                value: AttributeValue::Floats(all_target_weights),
            },
            AttributeProto {
                name: "n_targets".to_owned(),
                value: AttributeValue::Int(1),
            },
            AttributeProto {
                name: "aggregate_function".to_owned(),
                value: AttributeValue::String("SUM".to_owned()),
            },
            AttributeProto {
                name: "base_values".to_owned(),
                value: AttributeValue::Floats(vec![init]),
            },
            AttributeProto {
                name: "post_transform".to_owned(),
                value: AttributeValue::String("NONE".to_owned()),
            },
        ],
    };

    let graph = OnnxGraph {
        name: "GradientBoostingRegressor".to_owned(),
        nodes: vec![node],
        inputs: vec![ValueInfo {
            name: "X".to_owned(),
            elem_type: ONNX_FLOAT,
            shape: vec![-1, n_features as i64],
        }],
        outputs: vec![ValueInfo {
            name: "Y".to_owned(),
            elem_type: ONNX_FLOAT,
            shape: vec![-1, 1],
        }],
        initializers: vec![],
    };

    Ok(make_onnx_model(graph))
}

// --- GradientBoostingClassifier ---

impl ToOnnx for ferrolearn_tree::FittedGradientBoostingClassifier<f64> {
    fn to_onnx(&self) -> Result<OnnxModel, FerroError> {
        gradient_boosting_classifier_to_onnx(self)
    }
}

/// Build an ONNX model for a gradient boosting classifier.
fn gradient_boosting_classifier_to_onnx(
    model: &ferrolearn_tree::FittedGradientBoostingClassifier<f64>,
) -> Result<OnnxModel, FerroError> {
    let n_features = model.n_features();
    let classes = model.classes();
    let n_classes = classes.len();
    let lr = model.learning_rate() as f32;
    let init_values: Vec<f32> = model.init().iter().map(|&v| v as f32).collect();

    let mut all_treeids = Vec::new();
    let mut all_nodeids = Vec::new();
    let mut all_featureids = Vec::new();
    let mut all_values = Vec::new();
    let mut all_modes = Vec::new();
    let mut all_truenodeids = Vec::new();
    let mut all_falsenodeids = Vec::new();
    let mut all_class_treeids = Vec::new();
    let mut all_class_nodeids = Vec::new();
    let mut all_class_ids = Vec::new();
    let mut all_class_weights = Vec::new();

    // The trees are organized as trees[class_idx][round_idx] = Vec<Node>
    let mut global_tree_id = 0_i64;
    for (class_idx, class_trees) in model.trees().iter().enumerate() {
        for tree in class_trees {
            for (idx, node) in tree.iter().enumerate() {
                all_treeids.push(global_tree_id);
                all_nodeids.push(idx as i64);

                match node {
                    Node::Split {
                        feature,
                        threshold,
                        left,
                        right,
                        ..
                    } => {
                        all_featureids.push(*feature as i64);
                        all_values.push(*threshold as f32);
                        all_modes.push("BRANCH_LEQ".to_owned());
                        all_truenodeids.push(*left as i64);
                        all_falsenodeids.push(*right as i64);
                    }
                    Node::Leaf { value, .. } => {
                        all_featureids.push(0);
                        all_values.push(0.0);
                        all_modes.push("LEAF".to_owned());
                        all_truenodeids.push(0);
                        all_falsenodeids.push(0);

                        all_class_treeids.push(global_tree_id);
                        all_class_nodeids.push(idx as i64);
                        all_class_ids.push(class_idx as i64);
                        all_class_weights.push(*value as f32 * lr);
                    }
                }
            }
            global_tree_id += 1;
        }
    }

    let class_labels: Vec<i64> = classes.iter().map(|&c| c as i64).collect();

    let post_transform = if n_classes == 2 {
        "LOGISTIC"
    } else {
        "SOFTMAX"
    };

    let node = OnnxNode {
        inputs: vec!["X".to_owned()],
        outputs: vec!["label".to_owned(), "probabilities".to_owned()],
        op_type: "TreeEnsembleClassifier".to_owned(),
        domain: "ai.onnx.ml".to_owned(),
        name: "gradient_boosting_classifier".to_owned(),
        attributes: vec![
            AttributeProto {
                name: "nodes_treeids".to_owned(),
                value: AttributeValue::Ints(all_treeids),
            },
            AttributeProto {
                name: "nodes_nodeids".to_owned(),
                value: AttributeValue::Ints(all_nodeids),
            },
            AttributeProto {
                name: "nodes_featureids".to_owned(),
                value: AttributeValue::Ints(all_featureids),
            },
            AttributeProto {
                name: "nodes_values".to_owned(),
                value: AttributeValue::Floats(all_values),
            },
            AttributeProto {
                name: "nodes_modes".to_owned(),
                value: AttributeValue::Strings(all_modes),
            },
            AttributeProto {
                name: "nodes_truenodeids".to_owned(),
                value: AttributeValue::Ints(all_truenodeids),
            },
            AttributeProto {
                name: "nodes_falsenodeids".to_owned(),
                value: AttributeValue::Ints(all_falsenodeids),
            },
            AttributeProto {
                name: "class_treeids".to_owned(),
                value: AttributeValue::Ints(all_class_treeids),
            },
            AttributeProto {
                name: "class_nodeids".to_owned(),
                value: AttributeValue::Ints(all_class_nodeids),
            },
            AttributeProto {
                name: "class_ids".to_owned(),
                value: AttributeValue::Ints(all_class_ids),
            },
            AttributeProto {
                name: "class_weights".to_owned(),
                value: AttributeValue::Floats(all_class_weights),
            },
            AttributeProto {
                name: "classlabels_int64s".to_owned(),
                value: AttributeValue::Ints(class_labels),
            },
            AttributeProto {
                name: "base_values".to_owned(),
                value: AttributeValue::Floats(init_values),
            },
            AttributeProto {
                name: "post_transform".to_owned(),
                value: AttributeValue::String(post_transform.to_owned()),
            },
        ],
    };

    let graph = OnnxGraph {
        name: "GradientBoostingClassifier".to_owned(),
        nodes: vec![node],
        inputs: vec![ValueInfo {
            name: "X".to_owned(),
            elem_type: ONNX_FLOAT,
            shape: vec![-1, n_features as i64],
        }],
        outputs: vec![
            ValueInfo {
                name: "label".to_owned(),
                elem_type: 7, // INT64
                shape: vec![-1],
            },
            ValueInfo {
                name: "probabilities".to_owned(),
                elem_type: ONNX_FLOAT,
                shape: vec![-1, n_classes as i64],
            },
        ],
        initializers: vec![],
    };

    Ok(make_onnx_model(graph))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrolearn_core::introspection::HasCoefficients;
    use ferrolearn_core::traits::{Fit, Predict};
    use ndarray::{Array1, Array2, array};

    #[test]
    fn test_varint_encoding() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 0);
        assert_eq!(buf, vec![0]);

        buf.clear();
        encode_varint(&mut buf, 1);
        assert_eq!(buf, vec![1]);

        buf.clear();
        encode_varint(&mut buf, 127);
        assert_eq!(buf, vec![127]);

        buf.clear();
        encode_varint(&mut buf, 128);
        assert_eq!(buf, vec![0x80, 0x01]);

        buf.clear();
        encode_varint(&mut buf, 300);
        assert_eq!(buf, vec![0xAC, 0x02]);
    }

    #[test]
    fn test_protobuf_tag_encoding() {
        let mut buf = Vec::new();
        // Field 1, wire type 0 (varint) -> tag = (1 << 3) | 0 = 8
        encode_tag(&mut buf, 1, WIRE_VARINT);
        assert_eq!(buf, vec![0x08]);

        buf.clear();
        // Field 2, wire type 2 (length-delimited) -> tag = (2 << 3) | 2 = 18
        encode_tag(&mut buf, 2, WIRE_LEN);
        assert_eq!(buf, vec![0x12]);
    }

    #[test]
    fn test_onnx_model_serialization_produces_bytes() {
        let model = make_onnx_model(OnnxGraph {
            name: "test".to_owned(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            initializers: vec![],
        });
        let bytes = save_onnx_bytes(&model).unwrap();
        // Should produce non-empty protobuf output
        assert!(!bytes.is_empty());
        // First byte should be a valid protobuf tag
        // Field 1 (ir_version), wire type 0 = tag 0x08
        assert_eq!(bytes[0], 0x08);
    }

    #[test]
    fn test_linear_regression_to_onnx() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y = array![5.0, 11.0, 17.0, 23.0, 29.0];

        let model = ferrolearn_linear::LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let onnx = fitted.to_onnx().unwrap();

        // Check graph structure
        assert_eq!(onnx.graph.name, "FittedLinearRegression");
        assert_eq!(onnx.graph.nodes.len(), 1);
        assert_eq!(onnx.graph.nodes[0].op_type, "LinearRegressor");
        assert_eq!(onnx.graph.nodes[0].domain, "ai.onnx.ml");

        // Check that coefficients are embedded
        let coef_attr = onnx.graph.nodes[0]
            .attributes
            .iter()
            .find(|a| a.name == "coefficients")
            .unwrap();
        if let AttributeValue::Floats(ref floats) = coef_attr.value {
            assert_eq!(floats.len(), 2);
        } else {
            panic!("expected Floats attribute for coefficients");
        }

        // Check input/output shapes
        assert_eq!(onnx.graph.inputs.len(), 1);
        assert_eq!(onnx.graph.inputs[0].shape, vec![-1, 2]);
        assert_eq!(onnx.graph.outputs.len(), 1);

        // Verify coefficients match
        let coefs = fitted.coefficients();
        if let AttributeValue::Floats(ref floats) = coef_attr.value {
            for (i, &f) in floats.iter().enumerate() {
                assert!(
                    (f - coefs[i] as f32).abs() < 1e-4,
                    "coefficient mismatch at index {i}: onnx={f}, model={}",
                    coefs[i]
                );
            }
        }

        // Serialize to bytes and verify non-empty
        let bytes = save_onnx_bytes(&onnx).unwrap();
        assert!(bytes.len() > 50);
    }

    #[test]
    fn test_ridge_to_onnx() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = ferrolearn_linear::Ridge::<f64>::new().with_alpha(0.01);
        let fitted = model.fit(&x, &y).unwrap();

        let onnx = fitted.to_onnx().unwrap();
        assert_eq!(onnx.graph.nodes[0].op_type, "LinearRegressor");
        let bytes = save_onnx_bytes(&onnx).unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_logistic_regression_to_onnx() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 5.0, 5.0, 5.5, 5.5, 6.0, 6.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = ferrolearn_linear::LogisticRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let onnx = fitted.to_onnx().unwrap();
        assert_eq!(onnx.graph.nodes[0].op_type, "LinearClassifier");
        assert_eq!(onnx.graph.outputs.len(), 2);

        let bytes = save_onnx_bytes(&onnx).unwrap();
        assert!(bytes.len() > 50);
    }

    #[test]
    fn test_decision_tree_regressor_to_onnx() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];

        let model = ferrolearn_tree::DecisionTreeRegressor::<f64>::new().with_max_depth(Some(3));
        let fitted = model.fit(&x, &y).unwrap();

        let onnx = fitted.to_onnx().unwrap();
        assert_eq!(onnx.graph.nodes[0].op_type, "TreeEnsembleRegressor");

        let bytes = save_onnx_bytes(&onnx).unwrap();
        assert!(bytes.len() > 50);
    }

    #[test]
    fn test_decision_tree_classifier_to_onnx() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = ferrolearn_tree::DecisionTreeClassifier::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let onnx = fitted.to_onnx().unwrap();
        assert_eq!(onnx.graph.nodes[0].op_type, "TreeEnsembleClassifier");
        assert_eq!(onnx.graph.outputs.len(), 2);

        let bytes = save_onnx_bytes(&onnx).unwrap();
        assert!(bytes.len() > 50);
    }

    #[test]
    fn test_random_forest_classifier_to_onnx() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = ferrolearn_tree::RandomForestClassifier::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();

        let onnx = fitted.to_onnx().unwrap();
        assert_eq!(onnx.graph.nodes[0].op_type, "TreeEnsembleClassifier");

        let bytes = save_onnx_bytes(&onnx).unwrap();
        assert!(bytes.len() > 100);
    }

    #[test]
    fn test_random_forest_regressor_to_onnx() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let model = ferrolearn_tree::RandomForestRegressor::<f64>::new()
            .with_n_estimators(3)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();

        let onnx = fitted.to_onnx().unwrap();
        assert_eq!(onnx.graph.nodes[0].op_type, "TreeEnsembleRegressor");

        // Check aggregate function is AVERAGE for random forest
        let agg_attr = onnx.graph.nodes[0]
            .attributes
            .iter()
            .find(|a| a.name == "aggregate_function")
            .unwrap();
        if let AttributeValue::String(ref s) = agg_attr.value {
            assert_eq!(s, "AVERAGE");
        } else {
            panic!("expected String attribute for aggregate_function");
        }

        let bytes = save_onnx_bytes(&onnx).unwrap();
        assert!(bytes.len() > 100);
    }

    #[test]
    fn test_gradient_boosting_regressor_to_onnx() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let model = ferrolearn_tree::GradientBoostingRegressor::<f64>::new()
            .with_n_estimators(10)
            .with_learning_rate(0.1)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();

        let onnx = fitted.to_onnx().unwrap();
        assert_eq!(onnx.graph.nodes[0].op_type, "TreeEnsembleRegressor");

        // Check base_values is present
        let base_attr = onnx.graph.nodes[0]
            .attributes
            .iter()
            .find(|a| a.name == "base_values")
            .unwrap();
        if let AttributeValue::Floats(ref vals) = base_attr.value {
            assert_eq!(vals.len(), 1);
        } else {
            panic!("expected Floats attribute for base_values");
        }

        let bytes = save_onnx_bytes(&onnx).unwrap();
        assert!(bytes.len() > 100);
    }

    #[test]
    fn test_gradient_boosting_classifier_to_onnx() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = ferrolearn_tree::GradientBoostingClassifier::<f64>::new()
            .with_n_estimators(10)
            .with_learning_rate(0.1)
            .with_random_state(42);
        let fitted = model.fit(&x, &y).unwrap();

        let onnx = fitted.to_onnx().unwrap();
        assert_eq!(onnx.graph.nodes[0].op_type, "TreeEnsembleClassifier");

        let bytes = save_onnx_bytes(&onnx).unwrap();
        assert!(bytes.len() > 100);
    }

    #[test]
    fn test_save_onnx_file() {
        let dir = tempdir();
        let path = dir.join("model.onnx");

        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];
        let model = ferrolearn_linear::LinearRegression::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let onnx = fitted.to_onnx().unwrap();
        save_onnx(&onnx, &path).unwrap();

        let contents = std::fs::read(&path).unwrap();
        assert!(!contents.is_empty());
        // First byte should be protobuf tag for ir_version
        assert_eq!(contents[0], 0x08);
    }

    #[test]
    fn test_linear_regression_f32_to_onnx() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let y = ndarray::Array1::from_vec(vec![2.0f32, 4.0, 6.0, 8.0]);
        let model = ferrolearn_linear::LinearRegression::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let onnx = fitted.to_onnx().unwrap();
        let bytes = save_onnx_bytes(&onnx).unwrap();
        assert!(!bytes.is_empty());
    }

    fn tempdir() -> std::path::PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let dir = std::env::temp_dir().join(format!("ferrolearn_onnx_test_{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
