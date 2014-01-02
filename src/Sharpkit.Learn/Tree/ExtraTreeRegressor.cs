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
    /*/// <seealso cref="ExtraTreesRegressor"/>
    /// <seealso cref="ExtraTreesClassifier"/>*/
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
        public ExtraTreeRegressor(
            Criterion criterion = Criterion.Mse,
            Splitter splitter = Splitter.Random,
            int? max_depth = null,
            int min_samples_split = 2,
            int min_samples_leaf = 1,
            MaxFeaturesChoice max_features = null,
            Random random = null) : base(criterion,
                                         splitter,
                                         max_depth,
                                         min_samples_split,
                                         min_samples_leaf,
                                         max_features ?? MaxFeaturesChoice.Auto(),
                                         random)
        {
        }
    }
}
