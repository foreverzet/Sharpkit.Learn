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
    /// <code>
    /// <![CDATA[
    ///   using Sharpkit.Learn;
    ///   using Sharpkit.Learn.Tree;
    ///   using Sharpkit.Learn.Datasets;
    /// 
    ///   var clf = new DecisionTreeClassifier<int>();
    ///   var iris = IrisDataset.Load();
    ///   clf.Fit(iris.Data, iris.Target);
    ///   var score = clf.Score(iris.Data, iris.Target);
    /// ]]>
    /// </code>
    /// </example>
    public class DecisionTreeClassifier<TLabel> : BaseDecisionTree<TLabel>, IClassifier<TLabel>
    {
        /// <summary>
        /// Initializes a new instance of the DecisionTreeClassifier class.
        /// </summary>
        /// <param name="criterion">The function to measure the quality of a split. Supported criteria are
        /// <see cref="Criterion.Gini"/> for the Gini impurity and <see cref="Criterion.Entropy"/>
        /// for the information gain.</param>
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
        public DecisionTreeClassifier(
            Criterion criterion = Criterion.Gini,
            Splitter splitter = Splitter.Best,
            int? maxDepth = null,
            int minSamplesSplit = 2,
            int minSamplesLeaf = 1,
            MaxFeaturesChoice maxFeatures = null,
            Random random = null)
            : base(criterion, splitter, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures, random)
        {
        }

        /// <summary>
        /// Gets ordered list of class labeled discovered int <see cref="Fit"/>.
        /// </summary>
        public TLabel[] Classes
        {
            get { return this.ClassesInternal.ToArray(); }
        }

        /// <summary>
        /// Fit the model according to the given training data.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Training vectors,
        /// where nSamples is the number of samples and nFeatures
        /// is the number of features.</param>
        /// <param name="y">[nSamples] Target class labels.</param>
        /// <param name="sampleWeight">Individual weights for each sample. Array with dimensions [nSamples].</param>
        public void Fit(Matrix<double> x, TLabel[] y, Vector<double> sampleWeight = null)
        {
            this.FitClassification(x, y, sampleWeight);
        }

        /// <summary>
        /// Perform classification on samples in X.
        /// For an one-class model, +1 or -1 is returned.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Samples.</param>
        /// <returns>[nSamples] Class labels for samples in <paramref name="x"/>.</returns>
        public TLabel[] Predict(Matrix<double> x)
        {
            return this.PredictClassification(x);
        }

        /// <summary>
        /// Calculates probability estimates.
        /// The returned estimates for all classes are ordered by the
        /// label of classes.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Samples.</param>
        /// <returns>
        /// [nSamples, nClasses]. The probability of the sample for each class in the model,
        /// where classes are ordered as they are in <see cref="IClassifier{TLabel}.Classes"/>.
        /// </returns>
        public Matrix<double> PredictProba(Matrix<double> x)
        {
            int nSamples = x.RowCount;
            int nFeatures = x.ColumnCount;

            CheckFitted();

            if (this.NFeatures != nFeatures)
            {
                var message = string.Format(
                    "Number of features of the model must " +
                    " match the input. Model n_features is {0} and " +
                    " input n_features is {1}",
                    this.NFeatures,
                    nFeatures);

                throw new ArgumentException(message);
            }

            var proba = this.Tree.predict(x)[0];

            proba = proba.SubMatrix(0, proba.RowCount, 0, (int)this.NClasses[0]);
            var normalizer = proba.SumOfEveryRow();
            normalizer.MapInplace(v => v == 0.0 ? 1.0 : v, true);
            return proba.DivColumnVector(normalizer);
        }
    }
}
