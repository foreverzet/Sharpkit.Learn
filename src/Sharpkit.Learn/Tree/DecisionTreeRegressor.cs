// -----------------------------------------------------------------------
// <copyright file="DecisionTreeRegressor.cs" company="Sharpkit.Learn">
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
    /// A tree regressor.
    /// </summary>
    /// <seealso cref="DecisionTreeClassifier{TLabel}"/>
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
    /// >>> from sklearn.datasets import load_boston
    /// >>> from sklearn.cross_validation import cross_val_score
    /// >>> from sklearn.tree import DecisionTreeRegressor
    /// >>> boston = load_boston()
    /// >>> regressor = DecisionTreeRegressor(random_state=0)
    /// >>> cross_val_score(regressor, boston.data, boston.target, cv=10)
    /// array([ 0.61..., 0.57..., -0.34..., 0.41..., 0.75...,
    ///        0.07..., 0.29..., 0.33..., -1.42..., -1.77...])
    /// </example>
    public class DecisionTreeRegressor : BaseDecisionTree<int>, IRegressor
    {
        /*
    Attributes
    ----------
    `tree_` : Tree object
        The underlying Tree object.


    `max_features_` : int,
        The infered value of max_features.
    */

        /// <summary>
        /// Initializes a new instance of the DecisionTreeRegressor class.
        /// </summary>
        /// <param name="criterion"> The function to measure the quality of a split. The only supported
        /// criterion is <see cref="Criterion.Mse"/> for the mean squared error.</param>
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
        public DecisionTreeRegressor(
            Criterion criterion = Criterion.Mse,
            Splitter splitter = Splitter.Best,
            int? max_depth = null,
            int min_samples_split = 2,
            int min_samples_leaf = 1,
            MaxFeaturesChoice max_features = null,
            Random random = null) : base(criterion,
                                         splitter,
                                         max_depth,
                                         min_samples_split,
                                         min_samples_leaf,
                                         max_features,
                                         random)
        {
        }

        public void Fit(Matrix<double> x, Matrix<double> y, Vector<double> sampleWeight = null)
        {
            this.fitRegression(x, y, sampleWeight);
        }

        public Matrix<double> Predict(Matrix<double> x)
        {
            return this.predictRegression(x);
        }
    }
}
