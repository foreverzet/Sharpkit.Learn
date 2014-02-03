// -----------------------------------------------------------------------
// <copyright file="LogisticRegressionTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using System.IO;
using System.Text;
using Liblinear;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Sharpkit.Learn.Datasets;
using Sharpkit.Learn.LinearModel;

namespace Sharpkit.Learn.Test.LinearModel
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    [TestClass]
    public class LogisticRegressionTest
    {
        [ClassInitialize]
        public static void ClassInitialize(TestContext t)
        {
            Linear.disableDebugOutput();
        }

        private Matrix X = DenseMatrix.OfArray(new double[,] { { -1, 0 }, { 0, 1 }, { 1, 1 } });
        private Matrix X_sp = SparseMatrix.OfArray(new double[,] {{-1, 0}, {0, 1}, {1, 1}});

        private int[] Y1 = new[] {0, 1, 1};
        private int[] Y2 = new[] {2, 1, 0};
        private IrisDataset iris = IrisDataset.Load();

        /// <summary>
        /// Check that the model is able to fit the classification data
        /// </summary>
        private void check_predictions(LogisticRegression<int> clf, Matrix<double> X, int[] y)
        {
            int nSamples = y.Length;
            int[] classes = y.Distinct().OrderBy(v => v).ToArray();
            int nClasses = classes.Length;

            clf.Fit(X, y);
            var predicted = clf.Predict(X);
            Assert.IsTrue(classes.SequenceEqual(clf.Classes));

            Assert.AreEqual(nSamples, predicted.Length);
            Assert.IsTrue(y.SequenceEqual(predicted));
                
            Matrix<double> probabilities = clf.PredictProba(X);
            Assert.AreEqual(Tuple.Create(nSamples, nClasses), probabilities.Shape());
            Assert.IsTrue(probabilities.SumOfEveryRow().AlmostEquals(DenseVector.Create(probabilities.RowCount, i => 1.0)));
            Assert.IsTrue(y.SequenceEqual(probabilities.ArgmaxColumns()));
        }
        
        /// <summary>
        /// Simple sanity check on a 2 classes dataset
        /// Make sure it predicts the correct result on simple datasets.
        /// </summary>
        [TestMethod]
        public void TestPredict2Classes()
        {
            check_predictions(new LogisticRegression<int>(random: new Random(0)), X, Y1);
            check_predictions(new LogisticRegression<int>(random: new Random(0)), X_sp, Y1);

            check_predictions(new LogisticRegression<int>(c : 100, random : new Random(0)), X, Y1);
            check_predictions(new LogisticRegression<int>(c : 100, random : new Random(0)), X_sp, Y1);

            check_predictions(new LogisticRegression<int>(fitIntercept: false, random : new Random(0)), X, Y1);
            check_predictions(new LogisticRegression<int>(fitIntercept: false, random :new Random(0)), X_sp, Y1);
        }
        /*
        public void test_error()
        {
            //"""Test for appropriate exception on errors"""
            assert_raises(ValueError, logistic.LogisticRegression(C = -1).fit, X, Y1)
        }
        */
        [TestMethod]
        public void TestPredict3Classes()
        {
            check_predictions(new LogisticRegression<int>(c: 10), X, Y2);
            check_predictions(new LogisticRegression<int>(c: 10), X_sp, Y2);
        }
        
        /// <summary>
        /// Test logistic regression with the iris dataset.
        /// </summary>
        [TestMethod]
        public void TestPredictIris()
        {
            int nSamples = iris.Data.RowCount;

            string[] target = iris.Target.Select(v => iris.TargetNames[v]).ToArray();
            var clf = new LogisticRegression<string>(c: iris.Data.RowCount);
            clf.Fit(iris.Data, target);
            Assert.IsTrue(target.Distinct().OrderBy(t => t).SequenceEqual(clf.Classes));

            var pred = clf.Predict(iris.Data);
            var matchingN = pred.Zip(target, Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
            Assert.IsTrue(1.0*matchingN/pred.Length > 0.95);

            var probabilities = clf.PredictProba(iris.Data);
            Assert.IsTrue(probabilities.SumOfEveryRow().AlmostEquals(DenseVector.Create(nSamples, i => 1.0)));

            pred = probabilities.RowEnumerator().Select(r => iris.TargetNames[r.Item2.MaximumIndex()]).ToArray();
            matchingN = pred.Zip(target, Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
            Assert.IsTrue(1.0 * matchingN / pred.Length > 0.95);
        }
        
        /// <summary>
        /// Test sparsify and densify members.
        /// </summary>
        [TestMethod]
        public void TestSparsify()
        {
            string[] target = iris.Target.Select(t => iris.TargetNames[t]).ToArray();
            var clf = new LogisticRegression<string>(random: new Random(0));
            clf.Fit(iris.Data, target);

            Matrix<double> predDD = clf.DecisionFunction(iris.Data);

            clf.Sparsify();
            Assert.IsInstanceOfType(clf.Coef, typeof(SparseMatrix));
            Matrix<double> predSD = clf.DecisionFunction(iris.Data);

            Matrix spData = SparseMatrix.OfMatrix(iris.Data);
            Matrix<double> predSS = clf.DecisionFunction(spData);

            clf.Densify();
            Matrix<double> predDS = clf.DecisionFunction(spData);

            Assert.IsTrue(predDD.AlmostEquals(predSD));
            Assert.IsTrue(predDD.AlmostEquals(predSS));
            Assert.IsTrue(predDD.AlmostEquals(predDS));
        }

        /// <summary>
        /// Test that an exception is raised on inconsistent input.
        /// </summary>
        [TestMethod]
        public void TestInconsistentInput()
        {
            // todo:
            /*var rng = new Random(0);
            X_ = rng.random_sample((5, 10))
            y_ = np.ones(X_.shape[0])
            y_[0] = 0

            clf = logistic.LogisticRegression(random_state = 0)

            //# Wrong dimensions for training data
            y_wrong = y_[:
            -1]
            assert_raises(ValueError, clf.fit, X, y_wrong)

            //# Wrong dimensions for test data
            assert_raises(ValueError, clf.fit(X_, y_).predict,
              
             * rng.random_sample((3, 12)))
             * */
        }

        /// <summary>
        /// Test that we can write to coef_ and intercept_.
        /// </summary>
        [TestMethod]
        public void TestWriteParameters()
        {
            var clf = new LogisticRegression<int>(random: new Random(0));
            clf.Fit(X, Y1);
            clf.Coef.MapInplace(v => 0);
            clf.Intercept.MapInplace(v => 0);
            Assert.IsTrue(clf.DecisionFunction(X).Column(0).AlmostEquals(new DenseVector(new double[]{0, 0, 0})));
        }
        
        /// <summary>
        /// Test proper NaN handling.
        /// </summary>
        [TestMethod]
        [Ignore()]
        [ExpectedException(typeof (ArgumentException))]
        public void TestNan()
        {
            var xnan = X.Clone();
            xnan[0, 1] = double.NaN;
            new LogisticRegression<int>(random: new Random(0)).Fit(xnan, Y1);
        }
        

        [TestMethod]
        public void TestLiblinearRandomState()
        {
            var classification = SampleGenerator.MakeClassification(nSamples: 20);
            var lr1 = new LogisticRegression<int>(random: new Random(0));
            lr1.Fit(classification.X, classification.Y);
            var lr2 = new LogisticRegression<int>(random: new Random(0));
            lr2.Fit(classification.X, classification.Y);
            Assert.IsTrue(lr1.Coef.AlmostEquals(lr2.Coef));
        }
    }
}
