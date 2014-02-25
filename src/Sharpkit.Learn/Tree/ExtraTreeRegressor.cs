// -----------------------------------------------------------------------
// <copyright file="ExtraTreeRegressor.cs" company="Sharpkit.Learn">
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

    /// <summary>
    /// An extremely randomized tree regressor.
    /// 
    /// Extra-trees differ from classic decision trees in the way they are built.
    /// When looking for the best split to separate the samples of a node into two
    /// groups, random splits are drawn for each of the `max_features` randomly
    /// selected features and the best split among those is chosen. When
    /// `max_features` is set 1, this amounts to building a totally random
    /// decision tree.
    /// 
    /// Warning: Extra-trees should only be used within ensemble methods.
    /// </summary>
    /// <seealso cref="ExtraTreeClassifier{TLabel}"/>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/tree.py
    /// </remarks> 
    /// <remarks>
    /// <para>References</para>
    ///   [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
    ///   Machine Learning, 63(1), 3-42, 2006.
    /// </remarks>
    public class ExtraTreeRegressor : DecisionTreeRegressor
    {
        /// <summary>
        /// Initializes a new instance of the ExtraTreeRegressor class.
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
        public ExtraTreeRegressor(
            Criterion criterion = Criterion.Mse,
            Splitter splitter = Splitter.Random,
            int? maxDepth = null,
            int minSamplesSplit = 2,
            int minSamplesLeaf = 1,
            MaxFeaturesChoice maxFeatures = null,
            Random random = null) : base(criterion,
                                         splitter,
                                         maxDepth,
                                         minSamplesSplit,
                                         minSamplesLeaf,
                                         maxFeatures ?? MaxFeaturesChoice.Auto(),
                                         random)
        {
        }
    }
}
