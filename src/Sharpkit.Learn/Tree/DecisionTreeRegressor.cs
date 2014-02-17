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
    /// <code>
    /// <![CDATA[
    ///   using Sharpkit.Learn;
    ///   using Sharpkit.Learn.Tree;
    ///   using Sharpkit.Learn.Datasets;
    ///
    ///   var reg = new DecisionTreeRegressor();
    ///   var boston = BostonDataset.Load();
    ///   reg.Fit(boston.Data, boston.Target);
    ///   var score = Sharpkit.Learn.Metrics.Metrics.MeanSquaredError(
    ///     boston.Target,
    ///     reg.Predict(boston.Data).Column(0));
    /// ]]>
    /// </code>
    /// </example>
    public class DecisionTreeRegressor : BaseDecisionTree<int>, IRegressor
    {
        /// <summary>
        /// Initializes a new instance of the DecisionTreeRegressor class.
        /// </summary>
        /// <param name="criterion"> The function to measure the quality of a split. The only supported
        /// criterion is <see cref="Criterion.Mse"/> for the mean squared error.</param>
        /// <param name="splitter">The strategy used to choose the split at each node. Supported
        /// strategies are <see cref="Splitter.Best"/> to choose the best split and <see cref="Splitter.Random"/> to choose
        /// the best random split.</param>
        /// <param name="maxDepth">The maximum depth of the tree. If <c>null</c>, then nodes are expanded until
        /// all leaves are pure or until all leaves contain less than
        /// <paramref name="minSamplesSplit"/> samples.</param>
        /// <param name="minSamplesSplit">The minimum number of samples required to split an internal node.</param>
        /// <param name="minSamplesLeaf">The minimum number of samples required to be at a leaf node.</param>
        /// <param name="maxFeatures">Number of features to consider when looking for the best split. If null - 
        /// then all features will be considered.</param>
        /// <param name="random">random number generator</param>
        public DecisionTreeRegressor(
            Criterion criterion = Criterion.Mse,
            Splitter splitter = Splitter.Best,
            int? maxDepth = null,
            int minSamplesSplit = 2,
            int minSamplesLeaf = 1,
            MaxFeaturesChoice maxFeatures = null,
            Random random = null) : base(
                criterion,
                splitter,
                maxDepth,
                minSamplesSplit,
                minSamplesLeaf,
                maxFeatures,
                random)
        {
        }

        /// <summary>
        /// Fit the model according to the given training data.
        /// </summary>
        /// <param name="x">
        /// Matrix with dimensions [nSamples, nFeatures].
        /// Training vectors, where nSamples is the number of samples
        /// and nFeatures is the number of features.</param>
        /// <param name="y">Vector with dimensions [nSamples, nTargets]. Target values.</param>
        /// <param name="sampleWeight">Individual weights for each sample. Array with dimensions [nSamples].</param>
        public void Fit(Matrix<double> x, Matrix<double> y, Vector<double> sampleWeight = null)
        {
            this.FitRegression(x, y, sampleWeight);
        }

        /// <summary>
        /// Predict target values for samples in <paramref name="x"/>.
        /// </summary>
        /// <param name="x">Array with dimensions [nSamples, nFeatures].</param>
        /// <returns>Returns predicted values. Array with dimensions [nSamples, nTargets].</returns>
        public Matrix<double> Predict(Matrix<double> x)
        {
            return this.PredictRegression(x);
        }
    }
}
