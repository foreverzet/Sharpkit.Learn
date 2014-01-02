// -----------------------------------------------------------------------
// <copyright file="DecisionTreeClassifier.cs" company="Sharpkit.Learn">
//  Authors: Gilles Louppe <g.louppe@gmail.com>
//           Peter Prettenhofer <peter.prettenhofer@gmail.com>
//           Brian Holt <bdholt1@gmail.com>
//           Noel Dawe <noel@dawe.me>
//           Satrajit Gosh <satrajit.ghosh@gmail.com>
//           Lars Buitinck <L.J.Buitinck@uva.nl>
//           Sergey Zyuzin
//  Licence: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Tree
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// A decision tree classifier.
    /// </summary>
    /// <seealso cref="DecisionTreeRegressor"/>
    /// <remarks>
    /// <para>References</para>
    /// <para>
    ///  [1] http://en.wikipedia.org/wiki/Decision_tree_learning
    /// </para>
    /// <para>
    ///  [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
    ///     and Regression Trees", Wadsworth, Belmont, CA, 1984.
    /// </para>
    /// <para>
    ///  [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
    ///      Learning", Springer, 2009.
    /// </para>
    /// <para>
    ///  [4] L. Breiman, and A. Cutler, "Random Forests",
    ///      http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    /// </para>
    /// <para>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/tree.py
    /// </para>
    /// </remarks> 
    /// <example>
    ///  Examples
    /// >>> from sklearn.datasets import load_iris
    /// >>> from sklearn.cross_validation import cross_val_score
    /// >>> from sklearn.tree import DecisionTreeClassifier
    /// >>> clf = DecisionTreeClassifier(random_state=0)
    /// >>> iris = load_iris()
    /// >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ///                             
    /// array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
    ///        0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    /// </example>
    public class DecisionTreeClassifier<TLabel> : BaseDecisionTree<TLabel>, IClassifier<TLabel>
    {
        /*

    Attributes
    ----------
    `tree_` : Tree object
        The underlying Tree object.


    `max_features_` : int,
        The infered value of max_features.


    `n_classes_` : int or list
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    """*/

        /// <summary>
        /// Initializes a new instance of the DecisionTreeClassifier class.
        /// </summary>
        /// <param name="criterion">The function to measure the quality of a split. Supported criteria are
        /// <see cref="Criterion.Gini"/> for the Gini impurity and <see cref="Criterion.Entropy"/>
        /// for the information gain.</param>
        /// <param name="splitter">The strategy used to choose the split at each node. Supported
        /// strategies are <see cref="Splitter.Best"/> to choose the best split and <see cref="Splitter.Random"/> to choose
        /// the best random split.</param>
        /// <param name="max_depth">The maximum depth of the tree. If <c>null</c>, then nodes are expanded until
        /// all leaves are pure or until all leaves contain less than
        /// <paramref name="min_samples_split"/> samples.</param>
        /// <param name="min_samples_split">The minimum number of samples required to split an internal node.</param>
        /// <param name="min_samples_leaf">The minimum number of samples required to be at a leaf node.</param>
        /// <param name="max_features">Number of features to consider when looking for the best split. If null - 
        /// then all features will be considered.</param>
        /// <param name="random">random number generator</param>
        public DecisionTreeClassifier(
            Criterion criterion = Criterion.Gini,
            Splitter splitter = Splitter.Best,
            int? max_depth = null,
            int min_samples_split = 2,
            int min_samples_leaf = 1,
            MaxFeaturesChoice max_features = null,
            Random random = null)
            : base(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features, random)
        {
        }

        /// <summary>
        /// [n_classes] or a list of such arrays.
        /// The classes labels.
        /// </summary>
        public TLabel[] Classes
        {
            get { return this.classes_.ToArray(); }
        }

        /// <summary>
        /// Fit the model according to the given training data.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Training vectors,
        /// where nSamples is the number of samples and nFeatures
        /// is the number of features.</param>
        /// <param name="y">[nSamples] Target class labels.</param>
        /// <returns>Reference to itself.</returns>
        public void Fit(Matrix<double> x, TLabel[] y, Vector<double> sampleWeight = null)
        {
            this.fitClassification(x, y, sampleWeight);
        }

        /// <summary>
        /// Perform classification on samples in X.
        /// For an one-class model, +1 or -1 is returned.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Samples.</param>
        /// <returns>[nSamples] Class labels for samples in <paramref name="x"/>.</returns>
        public TLabel[] Predict(Matrix<double> x)
        {
            return this.predictClassification(x);
        }

        /// <summary>
        /// Predict class probabilities of the input samples X.
        /// </summary>
        /// <param name="X">[n_samples, n_features] The input samples.</param>
        /// <returns>[n_samples, n_classes]
        ///    The class probabilities of the input samples. Classes are ordered
        ///    by arithmetical order.</returns>
        public Matrix<double> PredictProba(Matrix<double> X)
        {
            int n_samples = X.RowCount;
            int n_features = X.ColumnCount;

            CheckFitted();

            if (this.n_features_ != n_features)
            {
                throw new ArgumentException(string.Format("Number of features of the model must " +
                                                          " match the input. Model n_features is {0} and " +
                                                          " input n_features is {1}",
                                                          this.n_features_, n_features));
            }

            var proba = this.tree_.predict(X)[0];


            proba = proba.SubMatrix(0, proba.RowCount, 0, (int)this.n_classes_[0]);
            var normalizer = proba.SumOfEveryRow();
            normalizer.MapInplace(v => v == 0.0 ? 1.0 : v, true);
            return proba.DivColumnVector(normalizer);
        }
    }
}
