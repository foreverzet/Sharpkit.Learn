// -----------------------------------------------------------------------
// <copyright file="MetricsTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using Sharpkit.Learn.Metrics;

namespace Sharpkit.Learn.Test.Metrics
{
    using System;
    using System.Linq;
    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Sharpkit.Learn.Datasets;
    using Sharpkit.Learn.Svm;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    [TestClass]
    public class MetricsTest
    {
        /// <summary>
        /// Make some classification predictions on a toy dataset using a SVC
        ///
        /// If binary is True restrict to a binary classification problem instead of a
        /// multiclass classification problem
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="binary"></param>
        private static Tuple<int[], int[], Matrix<double>> MakePrediction(
            Matrix<double> x = null,
            int[] y = null,
            bool binary = false)
        {
            if (x == null && y == null)
            {
                // import some data to play with
                var dataset = IrisDataset.Load();
                x = dataset.Data;
                y = dataset.Target;
            }

            if (binary)
            {
                // restrict to a binary classification task
                x = x.RowsAt(y.Indices(v => v < 2));
                y = y.Where(v => v < 2).ToArray();
            }

            int nSamples = x.RowCount;
            int nFeatures = x.ColumnCount;
            var rng = new Random(37);
            int[] p = Shuffle(rng, Enumerable.Range(0, nSamples).ToArray());
            x = x.RowsAt(p);
            y = y.ElementsAt(p);
            var half = nSamples/2;

            // add noisy features to make the problem harder and avoid perfect results
            rng = new Random(0);
            x = x.HStack(DenseMatrix.CreateRandom(nSamples, 200, new Normal{RandomSource = rng}));

            // run classifier, get class probabilities and label predictions
            var clf = new Svc<int>(kernel: Kernel.Linear, probability: true);
            clf.Fit(x.SubMatrix(0, half, 0, x.ColumnCount), y.Take(half).ToArray());
            Matrix<double> probasPred = clf.PredictProba(x.SubMatrix(half, x.RowCount - half, 0, x.ColumnCount));

            if (binary)
            {
                // only interested in probabilities of the positive case
                // XXX: do we really want a special API for the binary case?
                probasPred = probasPred.SubMatrix(0, probasPred.RowCount, 1, 1);
            }

            var yPred = clf.Predict(x.SubMatrix(half, x.RowCount - half, 0, x.ColumnCount));
            var yTrue = y.Skip(half).ToArray();
            return Tuple.Create(yTrue, yPred, probasPred);
        }

        private static int[] Shuffle(Random random, int[] range)
        {
            var result = (int[])range.Clone();
            for (int i = 0; i < range.Length; i++)
            {
                int x = random.Next(range.Length);
                int y = random.Next(range.Length);
                var tmp = result[x];
                result[x] = result[y];
                result[y] = tmp;
            }

            return result;
        }

        /// <summary>
        /// Test Precision Recall and F1 Score for binary classification task
        /// </summary>
        [TestMethod]
        public void TestPrecisionRecallF1ScoreBinary()
        {
            var prediction = MakePrediction(binary: true);
            var yTrue = prediction.Item1;
            var yPred = prediction.Item2;

            // detailed measures for each class
            var r = Sharpkit.Learn.Metrics.Metrics.PrecisionRecallFScoreSupport(yTrue, yPred);
            Assert.IsTrue(r.Precision.AlmostEquals(new[] {0.61, 0.83}, 10e-2));
            Assert.IsTrue(r.Recall.AlmostEquals(new[] {0.8, 0.66}, 10e-2));
            Assert.IsTrue(r.FBetaScore.AlmostEquals(new[] {0.69, 0.74}, 10e-2));
            Assert.IsTrue(r.Support.SequenceEqual(new[] {20, 30}));
 
            // individual scoring function that can be used for grid search: in the
            // binary class case the score is the value of the measure for the positive
            // class (e.g. label == 1)
            var ps = Learn.Metrics.Metrics.PrecisionScoreAvg(yTrue, yPred);
            Assert.AreEqual(0.85, ps, 10e-2);
            //Assert.IsTrue(ps.AlmostEquals(new[]{0.85}, 10e-2));

            var rs = Learn.Metrics.Metrics.RecallScoreAvg(yTrue, yPred);
            Assert.AreEqual(0.68, rs, 10e-2);

            var fs = Learn.Metrics.Metrics.F1ScoreAvg(yTrue, yPred);
            Assert.AreEqual(0.76, fs, 10e-2);
        }

        /// <summary>
        /// Test precision, recall and F1 score behave with a single positive or
        /// negative class
        ///
        /// Such a case may occur with non-stratified cross-validation
        /// </summary>
        [TestMethod]
        public void TestPrecisionRecallFBinarySingleClass()
        {
            Assert.AreEqual(1.0, Learn.Metrics.Metrics.PrecisionScoreAvg(new[] {1, 1}, new[] {1, 1}));
            Assert.AreEqual(1.0, Learn.Metrics.Metrics.RecallScoreAvg(new[] {1, 1}, new[] {1, 1}));
            Assert.AreEqual(1.0, Learn.Metrics.Metrics.F1ScoreAvg(new[] {1, 1}, new[] {1, 1}));
            
            Assert.AreEqual(0.0, Learn.Metrics.Metrics.PrecisionScoreAvg(new[] {-1, -1}, new[] {-1, -1}));
            Assert.AreEqual(0.0, Learn.Metrics.Metrics.RecallScoreAvg(new[] {-1, -1}, new[] {-1, -1}));
            Assert.AreEqual(0.0, Learn.Metrics.Metrics.F1ScoreAvg(new[] {-1, -1}, new[] {-1, -1}));
        }
        
        [TestMethod]
        public void TestPrecisionRecallFscoreSupportErrors()
        {
            var r = MakePrediction(binary: true);
            var yTrue = r.Item1;
            var yPred = r.Item2;

            try
            {
                // Bad beta
                Learn.Metrics.Metrics.PrecisionRecallFScoreSupport(yTrue, yPred, beta: 0.0);
                Assert.Fail("ArgumentException was not thrown");
            }
            catch (ArgumentException)
            {
            }
            
            try
            {
                // Bad pos_label
                Learn.Metrics.Metrics.PrecisionRecallFScoreSupportAvg(yTrue, yPred, posLabel :2, average:AverageKind.Macro);
                Assert.Fail("ArgumentException was not thrown");
            }
            catch (ArgumentException)
            {
            }

            try
            {
                // Bad average option
                Learn.Metrics.Metrics.PrecisionRecallFScoreSupportAvg(
                    new[] {0, 1, 2},
                    new[] {0, 1, 2},
                    posLabel :2,
                    average: (AverageKind)100);

                Assert.Fail("ArgumentException was not thrown");
            }
            catch (ArgumentException)
            {
            }
        }

        /// <summary>
        /// Test Precision Recall and F1 Score for multiclass classification task
        /// </summary>
        [TestMethod]
        public void TestPrecisionRecallF1ScoreMulticlass()
        {
            var r = MakePrediction(binary: false);
            var yTrue = r.Item1;
            var yPred = r.Item2;

            // compute scores with default labels introspection
            var result = Learn.Metrics.Metrics.PrecisionRecallFScoreSupport(yTrue, yPred);
            Assert.IsTrue(result.Precision.AlmostEquals(new[] {0.81, 0.42, 0.72}, 10e-2));
            Assert.IsTrue(result.Recall.AlmostEquals(new[] {0.81, 0.52, 0.6}, 10e-2));
            Assert.IsTrue(result.FBetaScore.AlmostEquals(new[] {0.81, 0.47, 0.65}, 10e-2));
            Assert.IsTrue(result.Support.SequenceEqual(new[] {22, 23, 30}));
            
            // averaging tests
            var ps = Learn.Metrics.Metrics.PrecisionScoreAvg(yTrue, yPred, average: AverageKind.Micro);
            Assert.AreEqual(0.64, ps, 10e-2);

            var rs = Learn.Metrics.Metrics.RecallScoreAvg(yTrue, yPred, average: AverageKind.Micro);
            Assert.AreEqual(0.64, rs, 10e-2);

            var fs = Learn.Metrics.Metrics.F1ScoreAvg(yTrue, yPred, average: AverageKind.Micro);
            Assert.AreEqual(0.64, fs, 10e-2);


            ps = Learn.Metrics.Metrics.PrecisionScoreAvg(yTrue, yPred, average: AverageKind.Macro);
            Assert.AreEqual(0.65, ps, 10e-2);

            rs = Learn.Metrics.Metrics.RecallScoreAvg(yTrue, yPred, average: AverageKind.Macro);
            Assert.AreEqual(0.64, rs, 10e-2);

            fs = Learn.Metrics.Metrics.F1ScoreAvg(yTrue, yPred, average: AverageKind.Macro);
            Assert.AreEqual(0.64, fs, 10e-2);

            ps = Learn.Metrics.Metrics.PrecisionScoreAvg(yTrue, yPred, average: AverageKind.Weighted);
            Assert.AreEqual(0.65, ps, 10e-2);

            rs = Learn.Metrics.Metrics.RecallScoreAvg(yTrue, yPred, average: AverageKind.Weighted);
            Assert.AreEqual(0.64, rs, 10e-2);

            fs = Learn.Metrics.Metrics.F1ScoreAvg(yTrue, yPred, average: AverageKind.Weighted);
            Assert.AreEqual(0.64, fs, 10e-2);

            /*try
            {
                Learn.Metrics.Metrics.PrecisionScore(yTrue, yPred, average: "samples");
                Assert.Fail();
            }
            catch (ArgumentException)
            {
            }

            try
            {
                Learn.Metrics.Metrics.RecallScore(yTrue, yPred, average: "samples");
                Assert.Fail();
            }
            catch (ArgumentException)
            {
            }

            try
            {
                Learn.Metrics.Metrics.F1Score(yTrue, yPred, average: "samples");
                Assert.Fail();
            }
            catch (ArgumentException)
            {
            }

            try
            {
                Learn.Metrics.Metrics.FBetaScore(yTrue, yPred, average: "samples", beta:0.5);
                Assert.Fail();
            }
            catch (ArgumentException)
            {
            }
            */
            // same prediction but with and explicit label ordering
            result = Learn.Metrics.Metrics.PrecisionRecallFScoreSupport(
                yTrue,
                yPred,
                labels: new[] {0, 2, 1});

            Assert.IsTrue(result.Precision.AlmostEquals(new[] {0.81, 0.72, 0.42}, 10e-2));
            Assert.IsTrue(result.Recall.AlmostEquals(new[] {0.81, 0.6, 0.52}, 10e-2));
            Assert.IsTrue(result.FBetaScore.AlmostEquals(new[] {0.81, 0.65, 0.47}, 10e-2));
            Assert.IsTrue(result.Support.SequenceEqual(new[] {22, 30, 23}));
        }

        /// <summary>
        /// Test Precision Recall and F1 Score for multiclass classification task
        ///
        /// GH Issue #1296
        /// </summary>
        [TestMethod]
        public void TestPrecisionRecallF1ScoreMulticlassPosLabelNone()
        {
            // initialize data
            var yTrue = new int[] {0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1};
            var yPred = new int[] {1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1};

            // compute scores with default labels introspection
            var r = Learn.Metrics.Metrics.PrecisionRecallFScoreSupportAvg(
                yTrue,
                yPred,
                posLabel: null,
                average: AverageKind.Weighted);
        }

        /// <summary>
        /// Check that pathological cases do not bring NaNs.
        /// </summary>
        [TestMethod]
        public void TestZeroPrecisionRecall()
        {
            var yTrue = new int[] {0, 1, 2, 0, 1, 2};
            var yPred = new int[] {2, 0, 1, 1, 2, 0};

            Assert.AreEqual(
                0.0, 
                Learn.Metrics.Metrics.PrecisionScoreAvg(
                    yTrue,
                    yPred,
                    average: AverageKind.Weighted),
                10e-2);

            Assert.AreEqual(
                0.0, 
                Learn.Metrics.Metrics.RecallScoreAvg(
                    yTrue,
                    yPred,
                    average: AverageKind.Weighted),
                10e-2);
       
            Assert.AreEqual(
                0.0, 
                Learn.Metrics.Metrics.F1ScoreAvg(
                    yTrue,
                    yPred,
                    average: AverageKind.Weighted),
                10e-2);
        }
   
        /*
   /// <summary>
   /// Test loss functions.
   /// </summary>
   [TestMethod]
   public void test_losses()
   {
        var prediction = make_prediction(binary: true);
        var n_samples = prediction.Item1.Length;
       var n_classes = prediction.Item1.Distinct().Count();

    // Classification
    // --------------
    // with warnings.catch_warnings(record=True):
    // Throw deprecated warning
    // assert_equal(zero_one(y_true, y_pred), 11)

    //assert_almost_equal(zero_one_loss(y_true, y_pred), 11 / float(n_samples), 2)
    // assert_equal(zero_one_loss(y_true, y_pred, normalize=False), 11)

    // assert_almost_equal(zero_one_loss(y_true, y_true), 0.0, 2)
    // assert_almost_equal(hamming_loss(y_true, y_pred), 2 * 11. / (n_samples * n_classes), 2)

    //assert_equal(accuracy_score(y_true, y_pred), 1 - zero_one_loss(y_true, y_pred))

    //with warnings.catch_warnings(True):
    // Throw deprecated warning
    // assert_equal(zero_one_score(y_true, y_pred), 1 - zero_one_loss(y_true, y_pred))

    // Regression
    // ----------
    //assert_almost_equal(mean_squared_error(y_true, y_pred), 10.999 / n_samples, 2)
    //assert_almost_equal(mean_squared_error(y_true, y_true), 0.00, 2)

    // mean_absolute_error and mean_squared_error are equal because
    // it is a binary problem.
    //assert_almost_equal(mean_absolute_error(y_true, y_pred),
    //                    10.999 / n_samples, 2)
    //assert_almost_equal(mean_absolute_error(y_true, y_true), 0.00, 2)

    //assert_almost_equal(explained_variance_score(y_true, y_pred), 0.16, 2)
    //assert_almost_equal(explained_variance_score(y_true, y_true), 1.00, 2)
    //assert_equal(explained_variance_score([0, 0, 0], [0, 1, 1]), 0.0)

       //assert_almost_equal(0.12, Learn.Metrics.Metrics.R2Score(prediction.Item1, prediction.Item2), 10e-2);
    //assert_almost_equal(r2_score(y_true, y_true), 1.00, 2)
    //assert_equal(r2_score([0, 0, 0], [0, 0, 0]), 1.0)
    //assert_equal(r2_score([0, 0, 0], [0, 1, 1]), 0.0)
   }*/

/*
        [TestMethod]
        public void test_losses_at_limits()
        {
            // test limit cases
            assert_almost_equal(mean_squared_error([0.], [0.]), 0.00, 2)
            assert_almost_equal(mean_absolute_error([0.], [0.]), 0.00, 2)
            assert_almost_equal(explained_variance_score([0.], [0.]), 1.00, 2)
            assert_almost_equal(r2_score([0., 1], [0., 1]), 1.00, 2)
        }*/

        /*[TestMethod]
        public void test_r2_one_case_error()
        {
            // test whether r2_score raises error given one point
            assert_raises(ValueError, r2_score, [0], [0])
        }*/

        /*
        [TestMethod]
        public void test_precision_recall_f1_score_with_an_empty_prediction()
        {
            y_true_ll = [(1,), (0,), (2, 1,)]
            y_pred_ll = [tuple(), (3,), (2, 1)]

            lb = LabelBinarizer()
            lb.fit([range(4)])
            y_true_bi = lb.transform(y_true_ll)
            y_pred_bi = lb.transform(y_pred_ll)

            for y_true, y_pred in [(y_true_ll, y_pred_ll), (y_true_bi, y_pred_bi)]
            {
                // true_pos = [ 0.  1.  1.  0.]
                // false_pos = [ 0.  0.  0.  1.]
                // false_neg = [ 1.  1.  0.  0.]

                p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                             average=None)
                assert_array_almost_equal(p, [0.0, 1.0, 1.0, 0.0], 2)
                assert_array_almost_equal(r, [0.0, 0.5, 1.0, 0.0], 2)
                assert_array_almost_equal(f, [0.0, 1 / 1.5, 1, 0.0], 2)
                assert_array_almost_equal(s, [1, 2, 1, 0], 2)

                p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                             average="macro")
                assert_almost_equal(p, 0.5)
                assert_almost_equal(r, 1.5 / 4)
                assert_almost_equal(f, 2.5 / (4 * 1.5))
                assert_equal(s, None)

                p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                             average="micro")
                assert_almost_equal(p, 2 / 3)
                assert_almost_equal(r, 0.5)
                assert_almost_equal(f, 2 / 3 / (2 / 3 + 0.5))
                assert_equal(s, None)

                p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                             average="weighted")
                assert_almost_equal(p, 3 / 4)
                assert_almost_equal(r, 0.5)
                assert_almost_equal(f, (2 / 1.5 + 1) / 4)
                assert_equal(s, None)

                p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                             average="samples")
                // |h(x_i) inter y_i | = [0, 0, 2]
                // |y_i| = [1, 1, 2]
                // |h(x_i)| = [0, 1, 2]
                assert_almost_equal(p, 1 / 3)
                assert_almost_equal(r, 2 / 3)
                assert_almost_equal(f, 1 / 3)
                assert_equal(s, None)
                }
        }
        */
/*
        [TestMethod]
        public void test_precision_recall_f1_no_labels()
        {
            y_true = np.zeros((20, 3))
            y_pred = np.zeros_like(y_true)

            p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                         average=None)
            //tp = [0, 0, 0]
            //fn = [0, 0, 0]
            //fp = [0, 0, 0]
            //support = [0, 0, 0]

            // Check per class
            assert_array_almost_equal(p, [0, 0, 0], 2)
            assert_array_almost_equal(r, [0, 0, 0], 2)
            assert_array_almost_equal(f, [0, 0, 0], 2)
            assert_array_almost_equal(s, [0, 0, 0], 2)

            // Check macro
            p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                         average="macro")
            assert_almost_equal(p, 0)
            assert_almost_equal(r, 0)
            assert_almost_equal(f, 0)
            assert_equal(s, None)

            // Check micro
            p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                         average="micro")
            assert_almost_equal(p, 0)
            assert_almost_equal(r, 0)
            assert_almost_equal(f, 0)
            assert_equal(s, None)

            // Check weighted
            p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                         average="weighted")
            assert_almost_equal(p, 0)
            assert_almost_equal(r, 0)
            assert_almost_equal(f, 0)
            assert_equal(s, None)

            // # Check example
            // |h(x_i) inter y_i | = [0, 0, 0]
            // |y_i| = [0, 0, 0]
            // |h(x_i)| = [1, 1, 2]
            p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                         average="samples")
            assert_almost_equal(p, 1)
            assert_almost_equal(r, 1)
            assert_almost_equal(f, 1)
            assert_equal(s, None)
        }
        */
    }
}
