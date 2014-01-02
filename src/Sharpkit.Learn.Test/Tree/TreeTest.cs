// -----------------------------------------------------------------------
// <copyright file="TreeTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Test.Tree
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Double;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Sharpkit.Learn.Datasets;
    using Sharpkit.Learn.Tree;

    /// <summary>
    /// Tests for Decition trees.
    /// </summary>
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

        public DecisionTreeClassifier<TLabel> CreateTree<TLabel>(
            string name,
            Criterion criterion = Criterion.Gini,
            Splitter splitter = Splitter.Best,
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
                    return new DecisionTreeClassifier<TLabel>(criterion, splitter, max_depth, min_samples_split,
                                                              min_samples_leaf, max_features, random);
                case "Presort-DecisionTreeClassifier":
                    return new DecisionTreeClassifier<TLabel>(criterion, Splitter.PresortBest, max_depth,
                                                              min_samples_split, min_samples_leaf, max_features, random);
                case "ExtraTreeClassifier":
                    return new ExtraTreeClassifier<TLabel>(criterion, splitter, max_depth, min_samples_split,
                                                           min_samples_leaf, max_features, random);
            }

            throw new InvalidOperationException("Unexpected name");
        }

        public DecisionTreeRegressor CreateRegressionTree(
            string name,
            Criterion criterion = Criterion.Mse,
            Splitter splitter = Splitter.Best,
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
                    return new DecisionTreeRegressor(criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
                                                     max_features, random);
                case "Presort-DecisionTreeRegressor":
                    return new DecisionTreeRegressor(criterion, Splitter.PresortBest, max_depth, min_samples_split,
                                                     min_samples_leaf, max_features, random);
                case "ExtraTreeRegressor":
                    return new ExtraTreeRegressor(criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
                                                  max_features, random);
            }

            throw new InvalidOperationException("Unexpected name");
        }


        private string[] REG_TREES = new[]
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
        private double[,] X = new double[,] {{-2, -1}, {-1, -1}, {-1, -2}, {1, 1}, {1, 2}, {2, 1}};
        private double[] y = new double[] {-1, -1, -1, 1, 1, 1};
        private double[,] T = new double[,] {{-1, -1}, {2, 2}, {3, 2}};

        private double[] true_result = new double[] {-1, 1, 1};


        private IrisDataset iris = IrisDataset.Load();
        private BostonDataset boston = BostonDataset.Load();
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
                var clf = CreateTree<double>(name, random: new Random(0));
                clf.Fit(X, y);
                AssertExt.ArrayEqual(clf.Predict(T), true_result, "Failed with {0}".Frmt(name));


                clf = CreateTree<double>(name, max_features: MaxFeaturesChoice.Value(1), random: new Random(1));
                clf.Fit(X, y);
                AssertExt.ArrayEqual(clf.Predict(T), true_result, "Failed with {0}".Frmt(name));
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
                var clf = CreateTree<double>(name, random: new Random(0));

                clf.Fit(X, y, sampleWeight: DenseVector.Create(X.GetLength(0), i => 1.0).ToArray());
                AssertExt.ArrayEqual(clf.Predict(T), true_result,
                                     "Failed with {0}".Frmt(name));


                clf.Fit(X, y, sampleWeight: DenseVector.Create(X.GetLength(0), i => 0.5).ToArray());
                AssertExt.ArrayEqual(clf.Predict(T), true_result,
                                     "Failed with {0}".Frmt(name));
            }
        }


        /// <summary>
        /// Check regression on a toy dataset.
        /// </summary>
        [TestMethod]
        public void TestRegressionToy()
        {
            foreach (var name in REG_TREES)
            {
                var reg = CreateRegressionTree(name, random: new Random(1));
                reg.Fit(X, y);
                AssertExt.ArrayEqual(reg.PredictSingle(T), true_result,
                                     "Failed with {0}".Frmt(name));


                var clf = CreateRegressionTree(name, max_features: MaxFeaturesChoice.Value(1), random: new Random(1));
                clf.Fit(X, y);
                AssertExt.AlmostEqual(reg.PredictSingle(T), true_result,
                                      "Failed with {0}".Frmt(name));
            }
        }

        /*
/// <summary>
/// Check on a XOR problem
/// </summary>
public void  test_xor()
{
    y = np.zeros((10, 10))
    y[:5, :5] = 1
    y[5:, 5:] = 1


    gridx, gridy = np.indices(y.shape)


    X = np.vstack([gridx.ravel(), gridy.ravel()]).T
    y = y.ravel()


    foreach (var name in CLF_TREES)
    {
        var clf = CreateTree(random: new Random(0));
        clf.Fit(X, y);
        AssertExt.ArrayEqual(clf.Score(X, y), 1.0,
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
                    var clf = CreateTree<int>(name, criterion: criterion, random: new Random(0));
                    clf.Fit(iris.Data, iris.Target);
                    var score = clf.Score(iris.Data, iris.Target);
                    Assert.IsTrue(score > 0.9,
                                  "Failed with {0}, criterion = {1} and score = {2}".Frmt(name, criterion, score));

                    clf = CreateTree<int>(name, criterion: criterion, max_features: MaxFeaturesChoice.Value(2),
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
            foreach (var name in REG_TREES)
            {
                foreach (var criterion in REG_CRITERIONS)
                {
                    var reg = CreateRegressionTree(name, criterion: criterion, random: new Random(0));
                    reg.Fit(boston.Data, boston.Target);
                    var score = Sharpkit.Learn.Metrics.Metrics.MeanSquaredError(boston.Target,
                                                                                reg.Predict(boston.Data).Column(0));
                    Assert.IsTrue(score < 1,
                                  "Failed with {0}, criterion = {1} and score = {2}".Frmt(name, criterion, score));


                    // using fewer features reduces the learning ability of this tree,
                    // but reduces training time.
                    reg = CreateRegressionTree(name, criterion: criterion, max_features: MaxFeaturesChoice.Value(6),
                                               random: new Random(0));
                    reg.Fit(boston.Data, boston.Target);
                    score = Sharpkit.Learn.Metrics.Metrics.MeanSquaredError(boston.Target,
                                                                            reg.Predict(boston.Data).Column(0));
                    Assert.IsTrue(score < 2,
                                  "Failed with {0}, criterion = {1} and score = {2}".Frmt(name, criterion, score));
                }
            }
        }

        /*
        /// <summary>
        /// Predict probabilities using DecisionTreeClassifier.
        /// </summary>
        [TestMethod]
        public void TestProbability()
        {
            foreach (var name in CLF_TREES)
            {
                var clf = CreateTree<int>(name, max_depth: 1, max_features: MaxFeaturesChoice.Value(1),
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

            foreach (var name in REG_TREES)
            {
                var reg = CreateRegressionTree(name, max_depth: null, random: new Random(0));
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
                var clf = CreateTree<int>(name, random: new Random(0));
                clf.Fit(X, y);
                AssertExt.ArrayEqual(clf.Predict(X), y,
                                     "Failed with {0}".Frmt(name));
            }

            foreach (var name in REG_TREES)
            {
                var reg = CreateRegressionTree(name, random: new Random(0));
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
        public void test_numerical_stability()
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
            foreach (var name in REG_TREES)
            {
                var reg = CreateRegressionTree(name, random: new Random(0));
                reg.Fit(X, y);
                reg.Fit(X, y.Select(v => -v).ToArray());
                reg.Fit(X.ToDenseMatrix()*-1, y.ToDenseVector());
                reg.Fit(X.ToDenseMatrix()*-1, y.ToDenseVector()*-1);
            }
        }

        /// <summary>
        /// Check variable importances.
        /// </summary>
        public void test_importances()
        {
            var classification = SampleGenerator.MakeClassification(nSamples: 2000,
                                                                    nFeatures: 10,
                                                                    nInformative: 3,
                                                                    nRedundant: 0,
                                                                    nRepeated: 0,
                                                                    shuffle: false,
                                                                    randomState: new Random(0));


            foreach (var name in CLF_TREES)
            {
                var clf = CreateTree<int>(name, random: new Random(0));
                clf.Fit(classification.X, classification.Y);
                var importances = clf.FeatureImportances();
                int n_important = importances.Where(v => v > 0.1).Count();

                Assert.AreEqual(10, importances.Count, "Failed with {0}".Frmt(name));
                Assert.AreEqual(3, n_important, "Failed with {0}".Frmt(name));


                Matrix<double> X_new = clf.Transform(X, threshold: ThresholdChoice.Mean());
                Assert.IsTrue(0 < X_new.ColumnCount, "Failed with {0}".Frmt(name));
                Assert.IsTrue(X_new.ColumnCount < X.GetLength(1), "Failed with {0}".Frmt(name));
            }
        }

/*


def test_max_features():
    """Check max_features."""
    for name, TreeRegressor in REG_TREES.items():
        reg = TreeRegressor(max_features="auto")
        reg.fit(boston.data, boston.target)
        assert_equal(reg.max_features_, boston.data.shape[1])


    for name, TreeClassifier in CLF_TREES.items():
        clf = TreeClassifier(max_features="auto")
        clf.fit(iris.data, iris.target)
        assert_equal(clf.max_features_, 2)


    for name, TreeEstimator in ALL_TREES.items():
        est = TreeEstimator(max_features="sqrt")
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_,
                     int(np.sqrt(iris.data.shape[1])))


        est = TreeEstimator(max_features="log2")
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_,
                     int(np.log2(iris.data.shape[1])))


        est = TreeEstimator(max_features=1)
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_, 1)


        est = TreeEstimator(max_features=3)
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_, 3)


        est = TreeEstimator(max_features=0.5)
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_,
                     int(0.5 * iris.data.shape[1]))


        est = TreeEstimator(max_features=1.0)
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_, iris.data.shape[1])


        est = TreeEstimator(max_features=None)
        est.fit(iris.data, iris.target)
        assert_equal(est.max_features_, iris.data.shape[1])


        # use values of max_features that are invalid
        est = TreeEstimator(max_features=10)
        assert_raises(ValueError, est.fit, X, y)


        est = TreeEstimator(max_features=-1)
        assert_raises(ValueError, est.fit, X, y)


        est = TreeEstimator(max_features=0.0)
        assert_raises(ValueError, est.fit, X, y)


        est = TreeEstimator(max_features=1.5)
        assert_raises(ValueError, est.fit, X, y)


        est = TreeEstimator(max_features="foobar")
        assert_raises(ValueError, est.fit, X, y)




def test_error():
    """Test that it gives proper exception on deficient input."""
    for name, TreeEstimator in CLF_TREES.items():
        # predict before fit
        est = TreeEstimator()
        assert_raises(Exception, est.predict_proba, X)


        est.fit(X, y)
        X2 = [-2, -1, 1]  # wrong feature shape for sample
        assert_raises(ValueError, est.predict_proba, X2)


    for name, TreeEstimator in ALL_TREES.items():
        # Invalid values for parameters
        assert_raises(ValueError, TreeEstimator(min_samples_leaf=-1).fit, X, y)
        assert_raises(ValueError, TreeEstimator(min_samples_split=-1).fit,
                      X, y)
        assert_raises(ValueError, TreeEstimator(max_depth=-1).fit, X, y)
        assert_raises(ValueError, TreeEstimator(max_features=42).fit, X, y)


        # Wrong dimensions
        est = TreeEstimator()
        y2 = y[:-1]
        assert_raises(ValueError, est.fit, X, y2)


        # Test with arrays that are non-contiguous.
        Xf = np.asfortranarray(X)
        est = TreeEstimator()
        est.fit(Xf, y)
        assert_almost_equal(est.predict(T), true_result)


        # predict before fitting
        est = TreeEstimator()
        assert_raises(Exception, est.predict, T)


        # predict on vector with different dims
        est.fit(X, y)
        t = np.asarray(T)
        assert_raises(ValueError, est.predict, t[:, 1:])


        # wrong sample shape
        Xt = np.array(X).T


        est = TreeEstimator()
        est.fit(np.dot(X, Xt), y)
        assert_raises(ValueError, est.predict, X)


        clf = TreeEstimator()
        clf.fit(X, y)
        assert_raises(ValueError, clf.predict, Xt)


        */
/*
/// <summary>
/// Test if leaves contain more than leaf_count training examples
/// </summary>
public void test_min_samples_leaf()
{
    foreach (var name in CLF_TREES)
    {
        var est = CreateTree<double>(min_samples_leaf: 5, random: new Random(0));
        est.Fit(X, y);
        @out = est.tree_.apply(X);
        node_counts = np.bincount(@out);
        leaf_count = node_counts[node_counts != 0]  // drop inner nodes
        assert_greater(np.min(leaf_count), 4,
                       "Failed with {0}".Frmt(name));
    }

    foreach (var name in REG_TREES)
    {
        var est = CreateRegressionTree(min_samples_leaf: 5, random: new Random(0));
        est.Fit(X, y);
        @out = est.tree_.apply(X);
        node_counts = np.bincount(@out);
        leaf_count = node_counts[node_counts != 0]  // drop inner nodes
        assert_greater(np.min(leaf_count), 4,
                       "Failed with {0}".Frmt(name));
    }
}
*/
        /*

def test_pickle():
    """Check that tree estimator are pickable """
    for name, TreeClassifier in CLF_TREES.items():
        clf = TreeClassifier(random_state=0)
        clf.fit(iris.data, iris.target)
        score = clf.score(iris.data, iris.target)


        serialized_object = pickle.dumps(clf)
        clf2 = pickle.loads(serialized_object)
        assert_equal(type(clf2), clf.__class__)
        score2 = clf2.score(iris.data, iris.target)
        assert_equal(score, score2, "Failed to generate same score "
                                    "after pickling (classification) "
                                    "with {0}".format(name))


    for name, TreeRegressor in REG_TREES.items():
        reg = TreeRegressor(random_state=0)
        reg.fit(boston.data, boston.target)
        score = reg.score(boston.data, boston.target)


        serialized_object = pickle.dumps(reg)
        reg2 = pickle.loads(serialized_object)
        assert_equal(type(reg2), reg.__class__)
        score2 = reg2.score(boston.data, boston.target)
        assert_equal(score, score2, "Failed to generate same score "
                                    "after pickling (regression) "
                                    "with {0}".format(name))




def test_multioutput():
    """Check estimators on multi-output problems."""
    X = [[-2, -1],
         [-1, -1],
         [-1, -2],
         [1, 1],
         [1, 2],
         [2, 1],
         [-2, 1],
         [-1, 1],
         [-1, 2],
         [2, -1],
         [1, -1],
         [1, -2]]


    y = [[-1, 0],
         [-1, 0],
         [-1, 0],
         [1, 1],
         [1, 1],
         [1, 1],
         [-1, 2],
         [-1, 2],
         [-1, 2],
         [1, 3],
         [1, 3],
         [1, 3]]


    T = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_true = [[-1, 0], [1, 1], [-1, 2], [1, 3]]


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


    # toy regression problem
    for name, TreeRegressor in REG_TREES.items():
        reg = TreeRegressor(random_state=0)
        y_hat = reg.fit(X, y).predict(T)
        assert_almost_equal(y_hat, y_true)
        assert_equal(y_hat.shape, (4, 2))




def test_classes_shape():
    """Test that n_classes_ and classes_ have proper shape."""
    for name, TreeClassifier in CLF_TREES.items():
        # Classification, single output
        clf = TreeClassifier(random_state=0)
        clf.fit(X, y)


        assert_equal(clf.n_classes_, 2)
        assert_array_equal(clf.classes_, [-1, 1])


        # Classification, multi-output
        _y = np.vstack((y, np.array(y) * 2)).T
        clf = TreeClassifier(random_state=0)
        clf.fit(X, _y)
        assert_equal(len(clf.n_classes_), 2)
        assert_equal(len(clf.classes_), 2)
        assert_array_equal(clf.n_classes_, [2, 2])
        assert_array_equal(clf.classes_, [[-1, 1], [-2, 2]])




def test_unbalanced_iris():
    """Check class rebalancing."""
    unbalanced_X = iris.data[:125]
    unbalanced_y = iris.target[:125]
    sample_weight = balance_weights(unbalanced_y)


    for name, TreeClassifier in CLF_TREES.items():
        clf = TreeClassifier(random_state=0)
        clf.fit(unbalanced_X, unbalanced_y, sample_weight=sample_weight)
        assert_almost_equal(clf.predict(unbalanced_X), unbalanced_y)




def test_memory_layout():
    """Check that it works no matter the memory layout"""
    for (name, TreeEstimator), dtype in product(ALL_TREES.items(),
                                                [np.float64, np.float32]):
        est = TreeEstimator(random_state=0)


        # Nothing
        X = np.asarray(iris.data, dtype=dtype)
        y = iris.target
        assert_array_equal(est.fit(X, y).predict(X), y)


        # C-order
        X = np.asarray(iris.data, order="C", dtype=dtype)
        y = iris.target
        assert_array_equal(est.fit(X, y).predict(X), y)


        # F-order
        X = np.asarray(iris.data, order="F", dtype=dtype)
        y = iris.target
        assert_array_equal(est.fit(X, y).predict(X), y)


        # Contiguous
        X = np.ascontiguousarray(iris.data, dtype=dtype)
        y = iris.target
        assert_array_equal(est.fit(X, y).predict(X), y)


        # Strided
        X = np.asarray(iris.data[::3], dtype=dtype)
        y = iris.target[::3]
        assert_array_equal(est.fit(X, y).predict(X), y)




def test_sample_weight():
    """Check sample weighting."""
    # Test that zero-weighted samples are not taken into account
    X = np.arange(100)[:, np.newaxis]
    y = np.ones(100)
    y[:50] = 0.0


    sample_weight = np.ones(100)
    sample_weight[y == 0] = 0.0


    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y, sample_weight=sample_weight)
    assert_array_equal(clf.predict(X), np.ones(100))


    # Test that low weighted samples are not taken into account at low depth
    X = np.arange(200)[:, np.newaxis]
    y = np.zeros(200)
    y[50:100] = 1
    y[100:200] = 2
    X[100:200, 0] = 200


    sample_weight = np.ones(200)


    sample_weight[y == 2] = .51  # Samples of class '2' are still weightier
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X, y, sample_weight=sample_weight)
    assert_equal(clf.tree_.threshold[0], 149.5)


    sample_weight[y == 2] = .50  # Samples of class '2' are no longer weightier
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    clf.fit(X, y, sample_weight=sample_weight)
    assert_equal(clf.tree_.threshold[0], 49.5)  # Threshold should have moved


    # Test that sample weighting is the same as having duplicates
    X = iris.data
    y = iris.target


    duplicates = rng.randint(0, X.shape[0], 200)


    clf = DecisionTreeClassifier(random_state=1)
    clf.fit(X[duplicates], y[duplicates])


    sample_weight = bincount(duplicates, minlength=X.shape[0])
    clf2 = DecisionTreeClassifier(random_state=1)
    clf2.fit(X, y, sample_weight=sample_weight)


    internal = clf.tree_.children_left != tree._tree.TREE_LEAF
    assert_array_almost_equal(clf.tree_.threshold[internal],
                              clf2.tree_.threshold[internal])




def test_32bit_equality():
    """Check if 32bit and 64bit get the same result. """
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(boston.data,
                                                        boston.target,
                                                        random_state=1)
    est = DecisionTreeRegressor(random_state=1)


    est.fit(X_train, y_train)
    score = est.score(X_test, y_test)
    assert_almost_equal(0.84652100667116, score)

 
 
*/
    }
}
