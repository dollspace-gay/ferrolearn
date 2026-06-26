//! Green guards for tree visualization helpers against the shape of
//! `sklearn.tree.export_graphviz` / `sklearn.tree.plot_tree`.
//!
//! Live sklearn 1.5.2 oracle for the classifier fixture:
//!
//! ```text
//! X = [[0.0], [1.0], [2.0], [3.0]]
//! y = [0, 0, 1, 1]
//! DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
//! export_graphviz(..., feature_names=["x0"], class_names=["zero", "one"],
//!                 rounded=True, impurity=False, precision=2)
//! ```
//!
//! The oracle DOT contains one root split `x0 <= 1.5`, two leaves, root-edge
//! `True` / `False` labels, and class labels `zero` / `one`. sklearn's
//! `plot_tree` returns five matplotlib annotations on this fixture: three node
//! boxes plus two root-edge labels. ferrolearn returns renderer-agnostic node
//! annotations instead, with the same node labels and root-edge labels carried
//! as data.
//!
//! Remaining gaps: no matplotlib artists, no file-handle output, no exact
//! sklearn color gradients, no impurity labels, and split-node values are
//! reconstructed from leaves rather than stored as sklearn's full `tree_.value`.

use ferrolearn_core::Fit;
use ferrolearn_tree::{
    DecisionTreeClassifier, DecisionTreeRegressor, Node, TreePlotOptions, export_graphviz,
    plot_tree,
};
use ndarray::array;

#[test]
fn green_export_graphviz_classifier_stump_matches_sklearn_shape() {
    let x = array![[0.0], [1.0], [2.0], [3.0]];
    let y = array![0usize, 0, 1, 1];
    let fitted = DecisionTreeClassifier::<f64>::new()
        .with_max_depth(Some(1))
        .fit(&x, &y)
        .unwrap();

    let dot = export_graphviz(
        fitted.nodes(),
        Some(&["x0"]),
        Some(&["zero", "one"]),
        TreePlotOptions {
            precision: 2,
            rounded: true,
            ..TreePlotOptions::default()
        },
    )
    .unwrap();

    assert!(dot.starts_with("digraph Tree {\n"));
    assert!(dot.contains("node [shape=box, style=\"rounded\""));
    assert!(dot.contains("0 [label=\"x0 <= 1.5\\nsamples = 4\\nvalue = [2, 2]\\nclass = zero\"]"));
    assert!(dot.contains("1 [label=\"samples = 2\\nvalue = [2, 0]\\nclass = zero\"]"));
    assert!(dot.contains("2 [label=\"samples = 2\\nvalue = [0, 2]\\nclass = one\"]"));
    assert!(dot.contains("0 -> 1 [labeldistance=2.5, labelangle=45, headlabel=\"True\"]"));
    assert!(dot.contains("0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel=\"False\"]"));
    assert!(dot.ends_with('}'));
}

#[test]
fn green_plot_tree_classifier_annotations_are_positioned_and_labeled() {
    let x = array![[0.0], [1.0], [2.0], [3.0]];
    let y = array![0usize, 0, 1, 1];
    let fitted = DecisionTreeClassifier::<f64>::new()
        .with_max_depth(Some(1))
        .fit(&x, &y)
        .unwrap();

    let annotations = plot_tree(
        fitted.nodes(),
        Some(&["x0"]),
        Some(&["zero", "one"]),
        TreePlotOptions {
            precision: 2,
            node_ids: true,
            filled: true,
            ..TreePlotOptions::default()
        },
    )
    .unwrap();

    assert_eq!(annotations.len(), 3);
    assert_eq!(annotations[0].node_id, 0);
    assert_eq!(annotations[0].parent, None);
    assert!(annotations[0].label.contains("node #0"));
    assert!(annotations[0].label.contains("x0 <= 1.5"));
    assert_eq!(annotations[1].parent, Some(0));
    assert_eq!(annotations[1].edge_label.as_deref(), Some("True"));
    assert_eq!(annotations[2].parent, Some(0));
    assert_eq!(annotations[2].edge_label.as_deref(), Some("False"));
    for annotation in annotations {
        assert!((0.0..=1.0).contains(&annotation.x));
        assert!((0.0..=1.0).contains(&annotation.y));
        assert!(annotation.fill_color.is_some());
    }
}

#[test]
fn green_export_and_plot_tree_regressor_values_and_validation() {
    let x = array![[0.0], [1.0], [2.0], [3.0]];
    let y = array![1.0, 1.5, 10.0, 11.0];
    let fitted = DecisionTreeRegressor::<f64>::new()
        .with_max_depth(Some(1))
        .fit(&x, &y)
        .unwrap();

    let dot = export_graphviz(
        fitted.nodes(),
        Some(&["x0"]),
        None,
        TreePlotOptions {
            precision: 2,
            proportion: true,
            ..TreePlotOptions::default()
        },
    )
    .unwrap();
    assert!(dot.contains("x0 <="));
    assert!(dot.contains("samples = 100%"));
    assert!(dot.contains("value = ["));

    let annotations = plot_tree(
        fitted.nodes(),
        Some(&["x0"]),
        None,
        TreePlotOptions::default(),
    )
    .unwrap();
    assert_eq!(annotations.len(), fitted.nodes().len());

    assert!(export_graphviz::<f64>(&[], None, None, TreePlotOptions::default()).is_err());
    assert!(
        export_graphviz(
            &[Node::Split {
                feature: 1,
                threshold: 0.0,
                left: 1,
                right: 2,
                impurity_decrease: 1.0,
                n_samples: 2,
            }],
            Some(&["x0"]),
            None,
            TreePlotOptions::default()
        )
        .is_err()
    );
}
