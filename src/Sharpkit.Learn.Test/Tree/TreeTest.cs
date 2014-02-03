// -----------------------------------------------------------------------
// <copyright file="TreeTest.cs" company="Sharpkit.Learn">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Test.Tree
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Sharpkit.Learn.Datasets;
    using Sharpkit.Learn.Preprocessing;
    using Sharpkit.Learn.Tree;
    using Sharpkit.Learn.FeatureSelection;
    using Sharpkit.Learn.Utils;

    /// <summary>
    /// Tests for Decition trees.
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/7d9984b995c50442458d1e010d36799cebbb260f/sklearn/tree/tests/test_tree.py
    /// </remarks>
    [TestClass]
    public class TreeTest
    {
        private Criterion[] CLF_CRITERIONS = new[] {Criterion.Gini, Criterion.Entropy};
        private Criterion[] REG_CRITERIONS = new[] {Criterion.Mse};

        private string[] CLF_TREES = new[]
                                         {
                                             "DecisionTreeClassifier",
                                             "Presort-DecisionTreeClassifier",
                                             "ExtraTreeClassifier"
                                         };

        public DecisionTreeClassifier<TLabel> CreateClassifier<TLabel>(
            string name,
            Criterion criterion = Criterion.Gini,
            int? max_depth = null,
            int min_samples_split = 2,
            int min_samples_leaf = 1,
            MaxFeaturesChoice max_features = null,
            Random random = null
            )
        {
            switch (name)
            {
                case "DecisionTreeClassifier":
                    return new DecisionTreeClassifier<TLabel>(criterion, Splitter.Best, max_depth, min_samples_split,
                                                              min_samples_leaf, max_features, random);
                case "Presort-DecisionTreeClassifier":
                    return new DecisionTreeClassifier<TLabel>(criterion, Splitter.PresortBest, max_depth,
                                                              min_samples_split, min_samples_leaf, max_features, random);
                case "ExtraTreeClassifier":
                    return new ExtraTreeClassifier<TLabel>(criterion, Splitter.Random, max_depth, min_samples_split,
                                                           min_samples_leaf, max_features, random);
            }

            throw new InvalidOperationException("Unexpected name");
        }

        public DecisionTreeRegressor CreateRegressor(
            string name,
            Criterion criterion = Criterion.Mse,
            int? max_depth = null,
            int min_samples_split = 2,
            int min_samples_leaf = 1,
            MaxFeaturesChoice max_features = null,
            Random random = null
            )
        {
            switch (name)
            {
                case "DecisionTreeRegressor":
                    return new DecisionTreeRegressor(criterion, Splitter.Best, max_depth, min_samples_split, min_samples_leaf,
                                                     max_features, random);
                case "Presort-DecisionTreeRegressor":
                    return new DecisionTreeRegressor(criterion, Splitter.PresortBest, max_depth, min_samples_split,
                                                     min_samples_leaf, max_features, random);
                case "ExtraTreeRegressor":
                    return new ExtraTreeRegressor(criterion, Splitter.Random, max_depth, min_samples_split, min_samples_leaf,
                                                  max_features, random);
            }

            throw new InvalidOperationException("Unexpected name");
        }

        private readonly string[] RegTrees = new[]
                                         {
                                             "DecisionTreeRegressor",
                                             "Presort-DecisionTreeRegressor",
                                             "ExtraTreeRegressor"
                                         };

        /*
ALL_TREES = dict()
ALL_TREES.update(CLF_TREES)
ALL_TREES.update(REG_TREES)



*/
// toy sample
        private readonly double[,] X = new double[,] {{-2, -1}, {-1, -1}, {-1, -2}, {1, 1}, {1, 2}, {2, 1}};
        private readonly double[] y = new double[] {-1, -1, -1, 1, 1, 1};
        private readonly double[,] T = new double[,] {{-1, -1}, {2, 2}, {3, 2}};

        private readonly double[] trueResult = new double[] {-1, 1, 1};


        private readonly IrisDataset iris = IrisDataset.Load();
        private readonly BostonDataset boston = BostonDataset.Load();
        /*# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]*/

/*
# also load the boston dataset
# and randomly permute it
boston = datasets.load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]

        */


        /// <summary>
        /// Check classification on a toy dataset.
        /// </summary>
        [TestMethod]
        public void TestClassificationToy()
        {
            foreach (var name in CLF_TREES)
            {
                var clf = CreateClassifier<double>(name, random: new Random(0));
                clf.Fit(X, y);
                AssertExt.ArrayEqual(clf.Predict(T), trueResult, "Failed with {0}".Frmt(name));


                clf = CreateClassifier<double>(name, max_features: MaxFeaturesChoice.Value(1), random: new Random(1));
                clf.Fit(X, y);
                AssertExt.ArrayEqual(clf.Predict(T), trueResult, "Failed with {0}".Frmt(name));
            }
        }


        /// <summary>
        /// Check classification on a weighted toy dataset.
        /// </summary>
        [TestMethod]
        public void TestWeightedClassificationToy()
        {
            foreach (var name in CLF_TREES)
            {
                var clf = CreateClassifier<double>(name, random: new Random(0));

                clf.Fit(X, y, sampleWeight: DenseVector.Create(X.GetLength(0), i => 1.0).ToArray());
                AssertExt.ArrayEqual(clf.Predict(T), trueResult,
                                     "Failed with {0}".Frmt(name));


                clf.Fit(X, y, sampleWeight: DenseVector.Create(X.GetLength(0), i => 0.5).ToArray());
                AssertExt.ArrayEqual(clf.Predict(T), trueResult,
                                     "Failed with {0}".Frmt(name));
            }
        }


        /// <summary>
        /// Check regression on a toy dataset.
        /// </summary>
        [TestMethod]
        public void TestRegressionToy()
        {
            foreach (var name in RegTrees)
            {
                var reg = CreateRegressor(name, random: new Random(1));
                reg.Fit(X, y);
                AssertExt.ArrayEqual(reg.PredictSingle(T), trueResult,
                                     "Failed with {0}".Frmt(name));


                var clf = CreateRegressor(name, max_features: MaxFeaturesChoice.Value(1), random: new Random(1));
                clf.Fit(X, y);
                AssertExt.AlmostEqual(reg.PredictSingle(T), trueResult,
                                      "Failed with {0}".Frmt(name));
            }
        }

/*
/// <summary>
/// Check on a XOR problem
/// </summary>
[TestMethod]
public void  test_xor()
{
    var y = new DenseMatrix(10, 10);
    y.SetSubMatrix(0, 5, 0, 5, DenseMatrix.Create(5, 5, (i, j) => 1.0));
    y.SetSubMatrix(5, 5, 5, 5, DenseMatrix.Create(5, 5, (i, j) => 1.0));

    gridx, gridy = np.indices(y.shape)

    X = np.vstack([gridx.ravel(), gridy.ravel()]).T
    y = y.ravel()


    foreach (var name in CLF_TREES)
    {
        var clf = CreateTree<double>(name, random: new Random(0));
        clf.Fit(X, y);
        AssertExt.ArrayEqual(1.0, clf.Score(X, y),
                             "Failed with {0}".Frmt(name));


        clf = CreateTree(random: new Random(0), max_features: MaxFeaturesChoice.Value(1));
        clf.Fit(X, y);
        AssertExt.AlmostEqual(clf.Score(X, y), 1.0,
                  "Failed with {0}".Frmt(name));
    }
}
*/

        /// <summary>
        /// Check consistency on dataset iris.
        /// </summary>
        [TestMethod]
        public void TestIris()
        {
            foreach (var name in CLF_TREES)
            {
                foreach (var criterion in CLF_CRITERIONS)
                {
                    var clf = CreateClassifier<int>(name, criterion: criterion, random: new Random(0));
                    clf.Fit(iris.Data, iris.Target);
                    var score = clf.Score(iris.Data, iris.Target);
                    Assert.IsTrue(score > 0.9,
                                  "Failed with {0}, criterion = {1} and score = {2}".Frmt(name, criterion, score));

                    clf = CreateClassifier<int>(name, criterion: criterion, max_features: MaxFeaturesChoice.Value(2),
                                          random: new Random(0));
                    clf.Fit(iris.Data, iris.Target);
                    score = clf.Score(iris.Data, iris.Target);
                    Assert.IsTrue(score > 0.9,
                                  "Failed with {0}, criterion = {1} and score = {2}".Frmt(name, criterion, score));
                }
            }
        }


        /// <summary>
        /// Check consistency on dataset boston house prices.
        /// </summary>
        [TestMethod]
        public void TestBoston()
        {
            foreach (var name in RegTrees)
            {
                foreach (var criterion in REG_CRITERIONS)
                {
                    var reg = CreateRegressor(name, criterion: criterion, random: new Random(0));
                    reg.Fit(boston.Data, boston.Target);
                    var score = Sharpkit.Learn.Metrics.Metrics.MeanSquaredError(boston.Target,
                                                                                reg.Predict(boston.Data).Column(0));
                    Assert.IsTrue(score < 1,
                                  "Failed with {0}, criterion = {1} and score = {2}".Frmt(name, criterion, score));


                    // using fewer features reduces the learning ability of this tree,
                    // but reduces training time.
                    reg = CreateRegressor(name, criterion: criterion, max_features: MaxFeaturesChoice.Value(6),
                                               random: new Random(0));
                    reg.Fit(boston.Data, boston.Target);
                    score = Sharpkit.Learn.Metrics.Metrics.MeanSquaredError(boston.Target,
                                                                            reg.Predict(boston.Data).Column(0));
                    Assert.IsTrue(score < 2,
                                  "Failed with {0}, criterion = {1} and score = {2}".Frmt(name, criterion, score));
                }
            }
        }


        /// <summary>
        /// Predict probabilities using DecisionTreeClassifier.
        /// </summary>
        [TestMethod]
        public void TestProbability()
        {
            foreach (var name in CLF_TREES)
            {
                var clf = CreateClassifier<int>(name, max_depth: 1, max_features: MaxFeaturesChoice.Value(1),
                                          random: new Random(42));
                clf.Fit(iris.Data, iris.Target);


                var prob_predict = clf.PredictProba(iris.Data);
                AssertExt.AlmostEqual(prob_predict.SumOfEveryRow().ToArray(),
                                      DenseVector.Create(iris.Data.RowCount, i => 1.0).ToArray(),
                                      "Failed with {0}".Frmt(name));

                AssertExt.ArrayEqual(prob_predict.ArgmaxColumns(),
                                     clf.Predict(iris.Data),
                                     "Failed with {0}".Frmt(name));

                AssertExt.AlmostEqual(clf.PredictProba(iris.Data),
                                      clf.PredictLogProba(iris.Data).Exp(),
                                      "Failed with {0}".Frmt(name), 10E-8);
            }
        }

        /// <summary>
        /// Check the array representation.
        /// </summary>
        [TestMethod]
        public void TestArrayrepr()
        {
            // Check resize
            var X = DenseMatrix.OfColumns(10000, 1, new[] {Enumerable.Range(0, 10000).Select(v => (double)v)});

            var y = DenseVector.OfEnumerable(Enumerable.Range(0, 10000).Select(v => (double)v));

            foreach (var name in RegTrees)
            {
                var reg = CreateRegressor(name, max_depth: null, random: new Random(0));
                reg.Fit(X, y);
            }
        }


        /// <summary>
        /// Check when y is pure.
        /// </summary>
        [TestMethod]
        public void TestPureSet()
        {
            var X = new double[,] {{-2, -1}, {-1, -1}, {-1, -2}, {1, 1}, {1, 2}, {2, 1}};
            var y = new int[] {1, 1, 1, 1, 1, 1};

            foreach (var name in CLF_TREES)
            {
                var clf = CreateClassifier<int>(name, random: new Random(0));
                clf.Fit(X, y);
                AssertExt.ArrayEqual(clf.Predict(X), y,
                                     "Failed with {0}".Frmt(name));
            }

            foreach (var name in RegTrees)
            {
                var reg = CreateRegressor(name, random: new Random(0));
                reg.Fit(X, y.Select(v => (double)v).ToArray());
                AssertExt.AlmostEqual(reg.Predict(X).ToDenseMatrix().Column(0).ToArray(),
                                      y.Select(v => (double)v).ToArray(),
                                      "Failed with {0}".Frmt(name));
            }
        }

        /// <summary>
        /// Check numerical stability.
        /// </summary>
        [TestMethod]
        public void TestNumericalStability()
        {
            var X = new double[,]
                        {
                            {152.08097839, 140.40744019, 129.75102234, 159.90493774},
                            {142.50700378, 135.81935120, 117.82884979, 162.75781250},
                            {127.28772736, 140.40744019, 129.75102234, 159.90493774},
                            {132.37025452, 143.71923828, 138.35694885, 157.84558105},
                            {103.10237122, 143.71928406, 138.35696411, 157.84559631},
                            {127.71276855, 143.71923828, 138.35694885, 157.84558105},
                            {120.91514587, 140.40744019, 129.75102234, 159.90493774}
                        };


            var y = new[]
                        {1.0, 0.70209277, 0.53896582, 0.0, 0.90914464, 0.48026916, 0.49622521};


            //with np.errstate(all="raise"):
            foreach (var name in RegTrees)
            {
                var reg = CreateRegressor(name, random: new Random(0));
                reg.Fit(X, y);
                reg.Fit(X, y.Select(v => -v).ToArray());
                reg.Fit(X.ToDenseMatrix()*-1, y.ToDenseVector());
                reg.Fit(X.ToDenseMatrix()*-1, y.ToDenseVector()*-1);
            }
        }

        /// <summary>
        /// Check variable importances.
        /// </summary>
        [TestMethod]
        public void TestImportances()
        {
            var classification = SampleGenerator.MakeClassification(nSamples: 200,
                                                                    nFeatures: 10,
                                                                    nInformative: 3,
                                                                    nRedundant: 0,
                                                                    nRepeated: 0,
                                                                    shuffle: false,
                                                                    randomState: new Random(13));

            //var xstr = "[" + string.Join(",", classification.X.RowEnumerator().Select(r => "[" + string.Join(",", r.Item2) + "]")) + "]";
            //var ystr = "[" + string.Join(",", classification.Y) + "]";
            foreach (var name in CLF_TREES)
            {
                var clf = CreateClassifier<int>(name, random: new Random(0));
                clf.Fit(classification.X, classification.Y);
                var importances = clf.FeatureImportances();
                int n_important = importances.Where(v => v > 0.1).Count();

                Assert.AreEqual(10, importances.Count, "Failed with {0}".Frmt(name));
                Assert.AreEqual(3, n_important, "Failed with {0}".Frmt(name));


                var X_new = clf.Transform(classification.X, threshold: ThresholdChoice.Mean());
                Assert.IsTrue(0 < X_new.ColumnCount, "Failed with {0}".Frmt(name));
                Assert.IsTrue(X_new.ColumnCount < classification.X.ColumnCount, "Failed with {0}".Frmt(name));
            }
        }


        /// <summary>
        /// Check max_features.
        /// </summary>
        [TestMethod]
        public void TestMaxFeatures()
        {
            foreach (var name in RegTrees)
            {
                var reg = CreateRegressor(name, max_features: MaxFeaturesChoice.Auto());
                reg.Fit(boston.Data, boston.Target);
                Assert.AreEqual(boston.Data.ColumnCount, reg.max_features_);
            }


            foreach (var name in CLF_TREES)
            {
                var clf = CreateClassifier<int>(name, max_features: MaxFeaturesChoice.Auto());
                clf.Fit(iris.Data, iris.Target);
                Assert.AreEqual(2, clf.max_features_);
            }

            foreach (var name in CLF_TREES)
            {
                var est = CreateClassifier<int>(name, max_features: MaxFeaturesChoice.Sqrt());
                est.Fit(iris.Data, iris.Target);
                Assert.AreEqual(Math.Sqrt(iris.Data.ColumnCount), est.max_features_);


                est = CreateClassifier<int>(name, max_features: MaxFeaturesChoice.Log2());
                est.Fit(iris.Data, iris.Target);
                Assert.AreEqual(Math.Log(iris.Data.ColumnCount, 2), est.max_features_);

                est = CreateClassifier<int>(name, max_features: MaxFeaturesChoice.Value(1));
                est.Fit(iris.Data, iris.Target);
                Assert.AreEqual(1, est.max_features_);

                est = CreateClassifier<int>(name, max_features: MaxFeaturesChoice.Value(3));
                est.Fit(iris.Data, iris.Target);
                Assert.AreEqual(3, est.max_features_);

                est = CreateClassifier<int>(name, max_features: MaxFeaturesChoice.Fraction(0.5));
                est.Fit(iris.Data, iris.Target);
                Assert.AreEqual((int)(0.5*iris.Data.ColumnCount), est.max_features_);

                est = CreateClassifier<int>(name, max_features: MaxFeaturesChoice.Fraction(1.0));
                est.Fit(iris.Data, iris.Target);
                Assert.AreEqual(iris.Data.ColumnCount, est.max_features_);

                //est = CreateTree<int>(name, max_features: null);
                //est.Fit(iris.Data, iris.Target);
                //Assert.AreEqual(est.max_features_, iris.Data.ColumnCount);

                var y_ = y.Select(v => (int)v).ToArray();
                // use values of max_features that are invalid
                est = CreateClassifier<int>(name, max_features: MaxFeaturesChoice.Value(10));
                AssertExt.Raises<ArgumentException>(() => est.Fit(X, y_));

                est = CreateClassifier<int>(name, max_features: MaxFeaturesChoice.Value(-1));
                AssertExt.Raises<ArgumentException>(() => est.Fit(X, y_));

                est = CreateClassifier<int>(name, max_features: MaxFeaturesChoice.Fraction(0.0));
                AssertExt.Raises<ArgumentException>(() => est.Fit(X, y_));

                est = CreateClassifier<int>(name, max_features: MaxFeaturesChoice.Fraction(1.5));
                AssertExt.Raises<ArgumentException>(() => est.Fit(X, y_));
            }
        }

        /// <summary>
        /// Test that it gives proper exception on deficient input.
        /// </summary>
        [TestMethod]
        public void TestError()
        {
            foreach (var name in CLF_TREES)
            {
                // predict before fit
                var est = CreateClassifier<double>(name);
                AssertExt.Raises<InvalidOperationException>(() => est.PredictProba(X));

                est.Fit(X, y);
                var x2 = new double[] {-2, -1, 1}.ToColumnMatrix(); // wrong feature shape for sample
                AssertExt.Raises<ArgumentException>(() => est.PredictProba(x2));
            }

            foreach (var name in CLF_TREES)
            {
                // Invalid values for parameters
                AssertExt.Raises<ArgumentException>(() => CreateClassifier<double>(name, min_samples_leaf: -1).Fit(X, y));
                AssertExt.Raises<ArgumentException>(() => CreateClassifier<double>(name, min_samples_split: -1).Fit(X, y));
                AssertExt.Raises<ArgumentException>(() => CreateClassifier<double>(name, max_depth: -1).Fit(X, y));
                AssertExt.Raises<ArgumentException>(
                    () => CreateClassifier<double>(name, max_features: MaxFeaturesChoice.Value(42)).Fit(X, y));

                // Wrong dimensions
                var est = CreateClassifier<double>(name);

                var y2 = y.Subarray(0, y.Length - 1);
                AssertExt.Raises<ArgumentException>(() => est.Fit(X, y2));

                // predict before fitting
                est = CreateClassifier<double>(name);
                AssertExt.Raises<InvalidOperationException>(() => est.Predict(T));

                // predict on vector with different dims
                est.Fit(X, y);
                AssertExt.Raises<ArgumentException>(
                    () => est.Predict(T.Subarray(0, T.GetLength(0), 1, T.GetLength(1) - 1)));

                // wrong sample shape
                est = CreateClassifier<double>(name);
                est.Fit(X.ToDenseMatrix()*X.ToDenseMatrix().Transpose(), y);
                AssertExt.Raises<ArgumentException>(() => est.Predict(X));


                est = CreateClassifier<double>(name);
                est.Fit(X, y);
                AssertExt.Raises<ArgumentException>(() => est.Predict(X.ToDenseMatrix().Transpose()));
            }
        }

        /// <summary>
        /// Test if leaves contain more than leaf_count training examples
        /// </summary>
        [TestMethod]
        public void TestMinSamplesLeaf()
        {
            foreach (var name in CLF_TREES)
            {
                var est = CreateClassifier<double>(name, min_samples_leaf: 5, random: new Random(0));
                est.Fit(X, y);
                var @out = est.tree_.apply(X.ToDenseMatrix());
                var node_counts = Np.BinCount(@out.Select(v => (int)v).ToArray());
                var leaf_count = node_counts.Where(v => v != 0).ToList(); // drop inner nodes
                Assert.IsTrue(leaf_count.Min() > 4,
                              "Failed with {0}".Frmt(name));
            }

            foreach (var name in RegTrees)
            {
                var est = CreateRegressor(name, min_samples_leaf: 5, random: new Random(0));
                est.Fit(X, y);
                var @out = est.tree_.apply(X.ToDenseMatrix());
                var nodeCounts = Np.BinCount(@out.Select(v => (int)v).ToArray());
                var leafCount = nodeCounts.Where(v => v != 0).ToList(); // drop inner nodes
                Assert.IsTrue(leafCount.Min() > 4,
                              "Failed with {0}".Frmt(name));
            }
        }

        /// <summary>
        /// Check estimators on multi-output problems.
        /// </summary>
        [TestMethod]
        public void TestMultioutput()
        {
            var X = new double[,]
                        {
                            {-2, -1},
                            {-1, -1},
                            {-1, -2},
                            {1, 1},
                            {1, 2},
                            {2, 1},
                            {-2, 1},
                            {-1, 1},
                            {-1, 2},
                            {2, -1},
                            {1, -1},
                            {1, -2}
                        }
                ;


            var y = new double[,]
                        {
                            {-1, 0},
                            {-1, 0},
                            {-1, 0},
                            {1, 1},
                            {1, 1},
                            {1, 1},
                            {-1, 2},
                            {-1, 2},
                            {-1, 2},
                            {1, 3},
                            {1, 3},
                            {1, 3}
                        };


            var T = new double[,] {{-1, -1}, {1, 1}, {-1, 1}, {1, -1}};
            var yTrue = new double[,] {{-1, 0}, {1, 1}, {-1, 2}, {1, 3}};

            /*
    # toy classification problem
    for name, TreeClassifier in CLF_TREES.items():
        clf = TreeClassifier(random_state=0)
        y_hat = clf.fit(X, y).predict(T)
        assert_array_equal(y_hat, y_true)
        assert_equal(y_hat.shape, (4, 2))


        proba = clf.predict_proba(T)
        assert_equal(len(proba), 2)
        assert_equal(proba[0].shape, (4, 2))
        assert_equal(proba[1].shape, (4, 4))


        log_proba = clf.predict_log_proba(T)
        assert_equal(len(log_proba), 2)
        assert_equal(log_proba[0].shape, (4, 2))
        assert_equal(log_proba[1].shape, (4, 4))
            */

            // toy regression problem
            foreach (var name in RegTrees)
            {
                var reg = CreateRegressor(name, random: new Random());
                reg.Fit(X, y);
                var yHat = reg.Predict(T);
                AssertExt.AlmostEqual(yTrue, yHat);
                Assert.AreEqual(Tuple.Create(4, 2), yHat.Shape());
            }
        }

        /// <summary>
        /// Test that n_classes_ and classes_ have proper shape.
        /// </summary>
        [TestMethod]
        public void TestClassesShape()
        {
            foreach (var name in CLF_TREES)
            {
                // Classification, single output
                var clf = CreateClassifier<double>(name, random: new Random(0));
                clf.Fit(X, y);

                Assert.AreEqual(1, clf.n_classes_.Count);
                Assert.AreEqual(2U, clf.n_classes_[0]);
                AssertExt.ArrayEqual(new [] {-1.0, 1.0}, clf.Classes);
            }
        }


        /// <summary>
        /// Compute sample weights such that the class distribution of y becomes
        /// balanced.
        /// </summary>
        /// <param name="?"></param>
        private static double[] BalanceWeights(int[] y)
        {
            var encoder = new LabelEncoder<int>();
            y = encoder.FitTransform(y);
            var bins = Np.BinCount(y);


            var weights = bins.ElementsAt(y).Select(v => 1.0/v*bins.Min()).ToArray();
            return weights;
        }

        [TestMethod]
        public void TestBalanceWeights()
        {
            var weights = BalanceWeights(new[] {0, 0, 1, 1});
            AssertExt.ArrayEqual(new[] {1.0, 1.0, 1.0, 1.0}, weights);

            weights = BalanceWeights(new[] {0, 1, 1, 1, 1});
            AssertExt.ArrayEqual(new[] {1.0, 0.25, 0.25, 0.25, 0.25}, weights);

            weights = BalanceWeights(new[] {0, 0});
            AssertExt.ArrayEqual(new[] {1.0, 1.0}, weights);
        }

        /// <summary>
        /// Check class rebalancing.
        /// </summary>
        [TestMethod]
        public void TestUnbalancedIris()
        {
            var unbalancedX = iris.Data.RowsAt(Enumerable.Range(0, 125));
            var unbalancedY = iris.Target.ElementsAt(Enumerable.Range(0, 125));
            var sampleWeight = BalanceWeights(unbalancedY);


            foreach (var name in CLF_TREES)
            {
                var clf = CreateClassifier<int>(name, random: new Random(0));
                clf.Fit(unbalancedX, unbalancedY, sampleWeight: sampleWeight);
                AssertExt.ArrayEqual(unbalancedY, clf.Predict(unbalancedX));
            }
        }

        /// <summary>
        /// Check sample weighting.
        /// </summary>
        [TestMethod]
        public void TestSampleWeight()
        {
            // Test that zero-weighted samples are not taken into account
            var X = Enumerable.Range(0, 100).ToColumnMatrix();
            var y = Enumerable.Repeat(1, 100).ToArray();
            Array.Clear(y, 0, 50);

            var sampleWeight = Enumerable.Repeat(1, 100).ToVector();
            sampleWeight.SetSubVector(0, 50, Enumerable.Repeat(0, 50).ToVector());

            var clf = new DecisionTreeClassifier<int>(random: new Random(0));
            clf.Fit(X, y, sampleWeight: sampleWeight);
            AssertExt.ArrayEqual(clf.Predict(X), Enumerable.Repeat(1, 100).ToArray());

            // Test that low weighted samples are not taken into account at low depth
            X = Enumerable.Range(0, 200).ToColumnMatrix();
            y = new int[200];
            Array.Copy(Enumerable.Repeat(1, 50).ToArray(), 0, y, 50, 50);
            Array.Copy(Enumerable.Repeat(2, 100).ToArray(), 0, y, 100, 100);
            X.SetSubMatrix(100, 100, 0, 1, Enumerable.Repeat(200, 100).ToColumnMatrix());

            sampleWeight = Enumerable.Repeat(1, 200).ToVector();

            sampleWeight.SetSubVector(100, 100, Enumerable.Repeat(0.51, 100).ToVector());
            // Samples of class '2' are still weightier
            clf = new DecisionTreeClassifier<int>(max_depth: 1, random: new Random(0));
            clf.Fit(X, y, sampleWeight: sampleWeight);
            Assert.AreEqual(149.5, clf.tree_.Threshold[0]);

            sampleWeight.SetSubVector(100, 100, Enumerable.Repeat(0.50, 100).ToVector());
            // Samples of class '2' are no longer weightier
            clf = new DecisionTreeClassifier<int>(max_depth: 1, random: new Random(0));
            clf.Fit(X, y, sampleWeight: sampleWeight);
            Assert.AreEqual(49.5, clf.tree_.Threshold[0]); // Threshold should have moved


            // Test that sample weighting is the same as having duplicates
            X = iris.Data;
            y = iris.Target;

            var random = new Random(0);
            var duplicates = new int[200];
            for (int i = 0; i < duplicates.Length; i++)
            {
                duplicates[i] = random.Next(X.RowCount);
            }

            clf = new DecisionTreeClassifier<int>(random: new Random(1));
            clf.Fit(X.RowsAt(duplicates), y.ElementsAt(duplicates));


            sampleWeight = Np.BinCount(duplicates, minLength: X.RowCount).ToVector();
            var clf2 = new DecisionTreeClassifier<int>(random: new Random(1));
            clf2.Fit(X, y, sampleWeight: sampleWeight);


            var @internal = clf.tree_.ChildrenLeft.Indices(v => v != Tree._TREE_LEAF);
            AssertExt.AlmostEqual(clf.tree_.Threshold.ElementsAt(@internal),
                                  clf2.tree_.Threshold.ElementsAt(@internal));
        }


        /// <summary>
        /// Check if 32bit and 64bit get the same result.
        /// </summary>
        [TestMethod]
        public void Test32BitEquality()
        {
            var r = CrossValidation.train_test_split(new[]
                                                         {
                                                             boston.Data,
                                                             boston.Target.ToColumnMatrix()
                                                         },
                                                     random_state: new Random(1));

            Matrix<double> xTrain = r[0].Item1;
            Matrix<double> xTest = r[0].Item2;

            Vector<double> yTrain = r[1].Item1.Column(0);
            Vector<double> yTest = r[1].Item2.Column(0);
            var est = new DecisionTreeRegressor(random: new Random(1));

            //var xstr = string.Join("\n", xTrain.RowEnumerator().Select(v => string.Join(",", v.Item2) ));
            //var ystr = "[" + string.Join(",", yTrain) + "]";

            est.Fit(xTrain, yTrain);
            double score = est.Score(xTest, yTest);
            Assert.AreEqual(0.74776186837750824, score, 1E-10);
        }
    }
}
