// -----------------------------------------------------------------------
// <copyright file="LinearTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Liblinear.Test
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    [TestClass]
    public class LinearTest
    {
        private static Random random = new Random(12345);

        [ClassInitialize]
        public static void disableDebugOutput(TestContext context)
        {
            Linear.disableDebugOutput();
        }

        public static Model createRandomModel() {
            Model model = new Model();
            model.solverType = SolverType.getById(SolverType.L2R_LR);
            model.bias = 2;
            model.label = new int[] {1, int.MaxValue, 2};
            model.w = new double[model.label.Length * 300];
            for (int i = 0; i < model.w.Length; i++) {
                // precision should be at least 1e-4
                model.w[i] = Math.Round(random.NextDouble() * 100000.0) / 10000.0;
            }

            // force at least one value to be zero
            model.w[random.Next(model.w.Length)] = 0.0;
            model.w[random.Next(model.w.Length)] = -0.0;


            model.nr_feature = model.w.Length / model.label.Length - 1;
            model.nr_class = model.label.Length;
            return model;
        }


        public static Problem createRandomProblem(int numClasses) {
            Problem prob = new Problem();
            prob.bias = -1;
            prob.l = random.Next(100) + 1;
            prob.n = random.Next(100) + 1;
            prob.x = new Feature[prob.l][];
            prob.y = new double[prob.l];


            for (int i = 0; i < prob.l; i++) {


                prob.y[i] = random.Next(numClasses);


                ISet<int> randomNumbers = new HashSet<int>();
                int num = random.Next(prob.n) + 1;
                for (int j = 0; j < num; j++) {
                    randomNumbers.Add(random.Next(prob.n) + 1);
                }
                List<int> randomIndices = new List<int>(randomNumbers);
                randomIndices.Sort();


                prob.x[i] = new Feature[randomIndices.Count];
                for (int j = 0; j < randomIndices.Count; j++) {
                    prob.x[i][j] = new Feature(randomIndices[j], random.NextDouble());
                }
            }
            return prob;
        }


        /**
         * create a very simple problem and check if the clearly separated examples are recognized as such
         */
        [TestMethod]
        public void testTrainPredict() {
            Problem prob = new Problem();
            prob.bias = -1;
            prob.l = 4;
            prob.n = 4;
            prob.x = new Feature[4][];
            prob.x[0] = new Feature[2];
            prob.x[1] = new Feature[1];
            prob.x[2] = new Feature[1];
            prob.x[3] = new Feature[3];


            prob.x[0][0] = new Feature(1, 1);
            prob.x[0][1] = new Feature(2, 1);


            prob.x[1][0] = new Feature(3, 1);
            prob.x[2][0] = new Feature(3, 1);


            prob.x[3][0] = new Feature(1, 2);
            prob.x[3][1] = new Feature(2, 1);
            prob.x[3][2] = new Feature(4, 1);


            prob.y = new double[4];
            prob.y[0] = 0;
            prob.y[1] = 1;
            prob.y[2] = 1;
            prob.y[3] = 0;


            foreach (SolverType solver in SolverType.values())
            {
                for (double C = 0.1; C <= 100.0; C *= 1.2)
                {

                    // compared the behavior with the C version
                    if (C < 0.2) if (solver.getId() == SolverType.L1R_L2LOSS_SVC) continue;
                    if (C < 0.7) if (solver.getId() == SolverType.L1R_LR) continue;

                    if (solver.isSupportVectorRegression())
                    {
                        continue;
                    }

                    Parameter param = new Parameter(solver, C, 0.1, 0.1);
                    Model model = Linear.train(prob, param);

                    double[] featureWeights = model.getFeatureWeights();
                    if (solver.getId() == SolverType.MCSVM_CS)
                    {
                        Assert.AreEqual(8, featureWeights.Length);
                    }
                    else
                    {
                        Assert.AreEqual(4, featureWeights.Length);
                    }

                    int i = 0;
                    foreach (double value in prob.y)
                    {
                        double prediction = Linear.predict(model, prob.x[i]);
                        Assert.AreEqual(value, prediction, "prediction with solver " + solver.Name);
                        if (model.isProbabilityModel())
                        {
                            double[] estimates = new double[model.getNrClass()];
                            double probabilityPrediction = Linear.predictProbability(model, prob.x[i], estimates);
                            Assert.AreEqual(prediction, probabilityPrediction);
                            Assert.IsTrue(estimates[(int)probabilityPrediction] >= 1.0 / model.getNrClass());
                            double estimationSum = 0;
                            foreach (double estimate in estimates) {
                                estimationSum += estimate;
                            }
                            Assert.AreEqual(1.0, estimationSum, 0.001);
                        }

                        i++;
                    }
                }
            }
        }


        [TestMethod]
        public void testCrossValidation()
        {
            int numClasses = random.Next(10) + 1;

            Problem prob = createRandomProblem(numClasses);

            Parameter param = new Parameter(SolverType.getById(SolverType.L2R_LR), 10, 0.01);
            int nr_fold = 10;
            double[] target = new double[prob.l];
            Linear.crossValidation(prob, param, nr_fold, target);

            foreach (double clazz in target)
            {
                Assert.IsTrue(clazz >= 0);
                Assert.IsTrue(clazz <= numClasses);
            }
        }


        [TestMethod]
        public void testLoadSaveModel()
        {
            Model model = null;
            foreach (SolverType solverType in SolverType.values())
            {
                model = createRandomModel();
                model.solverType = solverType;

                var tempFile = Path.GetTempFileName();
                Linear.saveModel(new FileInfo(tempFile), model);
                Model loadedModel = Linear.loadModel(new FileInfo(tempFile));
                Assert.AreEqual(model, loadedModel);
            }
        }

        [TestMethod]
        public void testTrainUnsortedProblem()
        {
            Problem prob = new Problem();
            prob.bias = -1;
            prob.l = 1;
            prob.n = 2;
            prob.x = new Feature[4][];
            prob.x[0] = new Feature[2];

            prob.x[0][0] = new Feature(2, 1);
            prob.x[0][1] = new Feature(1, 1);

            prob.y = new double[4];
            prob.y[0] = 0;

            Parameter param = new Parameter(SolverType.getById(SolverType.L2R_LR), 10, 0.1);
            try {
                Linear.train(prob, param);
                Assert.Fail("ArgumentException expected");
            } catch (ArgumentException e) {
            
                Assert.IsTrue(e.Message.Contains("nodes"));
                Assert.IsTrue(e.Message.Contains("sorted"));
                Assert.IsTrue(e.Message.Contains("ascending"));
                Assert.IsTrue(e.Message.Contains("order"));
            }
        }

        [TestMethod]
        public void testTrainTooLargeProblem()
        {
            Problem prob = new Problem();
            prob.l = 1000;
            prob.n = 20000000;
            prob.x = new Feature[prob.l][];
            prob.y = new double[prob.l];
            for (int i = 0; i < prob.l; i++) {
                prob.x[i] = new Feature[] {};
                prob.y[i] = i;
            }

            foreach (SolverType solverType in SolverType.values()) {
                if (solverType.isSupportVectorRegression()) continue;
                Parameter param = new Parameter(solverType, 10, 0.1);
                try {
                    Linear.train(prob, param);
                    Assert.Fail("ArgumentException expected");
                } catch (ArgumentException e) {
                    Assert.IsTrue(e.Message.Contains("number of classes"));
                    Assert.IsTrue(e.Message.Contains("too large"));
                }
            }
        }

        [TestMethod]
        public void testPredictProbabilityWrongSolver() {
            Problem prob = new Problem();
            prob.l = 1;
            prob.n = 1;
            prob.x = new Feature[prob.l][];
            prob.y = new double[prob.l];
            for (int i = 0; i < prob.l; i++) {
                prob.x[i] = new Feature[] {};
                prob.y[i] = i;
            }

            SolverType solverType = SolverType.getById(SolverType.L2R_L1LOSS_SVC_DUAL);
            Parameter param = new Parameter(solverType, 10, 0.1);
            Model model = Linear.train(prob, param);
            try {
                Linear.predictProbability(model, prob.x[0], new double[1]);
                Assert.Fail("IllegalArgumentException expected");
            } catch (ArgumentException e) {
                Assert.AreEqual("probability output is only supported for logistic regression." //
                    + " This is currently only supported by the following solvers:" //
                    + " L2R_LR, L1R_LR, L2R_LR_DUAL", e.Message);
            }
        }

        [TestMethod]
        public void testRealloc()
        {
            int[] f = new [] {1, 2, 3};
            f = Linear.copyOf(f, 5);
            f[3] = 4;
            f[4] = 5;
            Assert.IsTrue(new [] {1, 2, 3, 4, 5}.SequenceEqual(f));
        }

        [TestMethod]
        public void testAtoi()
        {
            Assert.AreEqual(25, Linear.atoi("+25"));
            Assert.AreEqual(-345345 , Linear.atoi("-345345"));
            Assert.AreEqual(0, Linear.atoi("+0"));
            Assert.AreEqual(0, Linear.atoi("0"));
            Assert.AreEqual(int.MaxValue, Linear.atoi("2147483647"));
            Assert.AreEqual(int.MinValue, Linear.atoi("-2147483648"));
        }


        [TestMethod]
        [ExpectedException(typeof(FormatException))]
        public void testAtoiInvalidData()
        {
            Linear.atoi("+");
        }

        [TestMethod]
        [ExpectedException(typeof(FormatException))]
        public void testAtoiInvalidData2()
        {
            Linear.atoi("abc");
        }

        [TestMethod]
        [ExpectedException(typeof(FormatException))]
        public void testAtoiInvalidData3()
        {
            Linear.atoi(" ");
        }

        [TestMethod]
        public void testAtof()
        {
            Assert.AreEqual(25.0, Linear.atof("+25"));
            Assert.AreEqual(-25.12345678, Linear.atof("-25.12345678"), 1e-10);
            Assert.AreEqual(0.345345299, Linear.atof("0.345345299"), 1e-10);
        }

        [TestMethod]
        [ExpectedException(typeof(FormatException))]
        public void testAtofInvalidData() {
            Linear.atof("0.5t");
        }

            /*
        [TestMethod]
        public void testSaveModelWithIOException() {
            Model model = createRandomModel();


            Writer out = PowerMockito.mock(Writer.class);


            IOException ioException = new IOException("some reason");


            doThrow(ioException).when(out).flush();


            try {
                Linear.saveModel(out, model);
                fail("IOException expected");
            } catch (IOException e) {
                assertThat(e).isEqualTo(ioException);
            }


            verify(out).flush();
            verify(out, times(1)).close();
        }
            */

        /**
         * compared input/output values with the C version (1.51)
         *
         * <pre>
         * IN:
         * res prob.l = 4
         * res prob.n = 4
         * 0: (2,1) (4,1)
         * 1: (1,1)
         * 2: (3,1)
         * 3: (2,2) (3,1) (4,1)
         *
         * TRANSPOSED:
         *
         * res prob.l = 4
         * res prob.n = 4
         * 0: (2,1)
         * 1: (1,1) (4,2)
         * 2: (3,1) (4,1)
         * 3: (1,1) (4,1)
         * </pre>
         */
        [TestMethod]
        public void testTranspose() {
            Problem prob = new Problem();
            prob.bias = -1;
            prob.l = 4;
            prob.n = 4;
            prob.x = new Feature[4][];
            prob.x[0] = new Feature[2];
            prob.x[1] = new Feature[1];
            prob.x[2] = new Feature[1];
            prob.x[3] = new Feature[3];


            prob.x[0][0] = new Feature(2, 1);
            prob.x[0][1] = new Feature(4, 1);


            prob.x[1][0] = new Feature(1, 1);
            prob.x[2][0] = new Feature(3, 1);


            prob.x[3][0] = new Feature(2, 2);
            prob.x[3][1] = new Feature(3, 1);
            prob.x[3][2] = new Feature(4, 1);


            prob.y = new double[4];
            prob.y[0] = 0;
            prob.y[1] = 1;
            prob.y[2] = 1;
            prob.y[3] = 0;


            Problem transposed = Linear.transpose(prob);


            Assert.AreEqual(1, transposed.x[0].Length);
            Assert.AreEqual(2, transposed.x[1].Length);
            Assert.AreEqual(2, transposed.x[2].Length);
            Assert.AreEqual(2, transposed.x[3].Length);

            Assert.AreEqual(new Feature(2, 1), transposed.x[0][0]);

            Assert.AreEqual(new Feature(1, 1), transposed.x[1][0]);
        
            Assert.AreEqual(new Feature(4, 2), transposed.x[1][1]);

            Assert.AreEqual(new Feature(3, 1), transposed.x[2][0]);
            Assert.AreEqual(new Feature(4, 1), transposed.x[2][1]);

            Assert.AreEqual(new Feature(1, 1), transposed.x[3][0]);
            Assert.AreEqual(new Feature(4, 1), transposed.x[3][1]);

            Assert.IsTrue(prob.y.SequenceEqual(transposed.y));
        }


        /**
         *
         * compared input/output values with the C version (1.51)
         *
         * <pre>
         * IN:
         * res prob.l = 5
         * res prob.n = 10
         * 0: (1,7) (3,3) (5,2)
         * 1: (2,1) (4,5) (5,3) (7,4) (8,2)
         * 2: (1,9) (3,1) (5,1) (10,7)
         * 3: (1,2) (2,2) (3,9) (4,7) (5,8) (6,1) (7,5) (8,4)
         * 4: (3,1) (10,3)
         *
         * TRANSPOSED:
         *
         * res prob.l = 5
         * res prob.n = 10
         * 0: (1,7) (3,9) (4,2)
         * 1: (2,1) (4,2)
         * 2: (1,3) (3,1) (4,9) (5,1)
         * 3: (2,5) (4,7)
         * 4: (1,2) (2,3) (3,1) (4,8)
         * 5: (4,1)
         * 6: (2,4) (4,5)
         * 7: (2,2) (4,4)
         * 8:
         * 9: (3,7) (5,3)
         * </pre>
         */
        [TestMethod]
        public void testTranspose2() {
            Problem prob = new Problem();
            prob.bias = -1;
            prob.l = 5;
            prob.n = 10;
            prob.x = new Feature[5][];
            prob.x[0] = new Feature[3];
            prob.x[1] = new Feature[5];
            prob.x[2] = new Feature[4];
            prob.x[3] = new Feature[8];
            prob.x[4] = new Feature[2];


            prob.x[0][0] = new Feature(1, 7);
            prob.x[0][1] = new Feature(3, 3);
            prob.x[0][2] = new Feature(5, 2);


            prob.x[1][0] = new Feature(2, 1);
            prob.x[1][1] = new Feature(4, 5);
            prob.x[1][2] = new Feature(5, 3);
            prob.x[1][3] = new Feature(7, 4);
            prob.x[1][4] = new Feature(8, 2);


            prob.x[2][0] = new Feature(1, 9);
            prob.x[2][1] = new Feature(3, 1);
            prob.x[2][2] = new Feature(5, 1);
            prob.x[2][3] = new Feature(10, 7);


            prob.x[3][0] = new Feature(1, 2);
            prob.x[3][1] = new Feature(2, 2);
            prob.x[3][2] = new Feature(3, 9);
            prob.x[3][3] = new Feature(4, 7);
            prob.x[3][4] = new Feature(5, 8);
            prob.x[3][5] = new Feature(6, 1);
            prob.x[3][6] = new Feature(7, 5);
            prob.x[3][7] = new Feature(8, 4);


            prob.x[4][0] = new Feature(3, 1);
            prob.x[4][1] = new Feature(10, 3);


            prob.y = new double[5];
            prob.y[0] = 0;
            prob.y[1] = 1;
            prob.y[2] = 1;
            prob.y[3] = 0;
            prob.y[4] = 1;


            Problem transposed = Linear.transpose(prob);


            Assert.AreEqual(3, transposed.x[0].Length);
            Assert.AreEqual(2, transposed.x[1].Length);
            Assert.AreEqual(4, transposed.x[2].Length);
            Assert.AreEqual(2, transposed.x[3].Length);
            Assert.AreEqual(4, transposed.x[4].Length);
            Assert.AreEqual(1, transposed.x[5].Length);
            Assert.AreEqual(2, transposed.x[6].Length);
            Assert.AreEqual(2, transposed.x[7].Length);
            Assert.AreEqual(0, transposed.x[8].Length);
            Assert.AreEqual(2, transposed.x[9].Length);
              
            Assert.AreEqual(new Feature(1, 7), transposed.x[0][0]);
            Assert.AreEqual(new Feature(3, 9), transposed.x[0][1]);
            Assert.AreEqual(new Feature(4, 2), transposed.x[0][2]);
        
            Assert.AreEqual(new Feature(2, 1), transposed.x[1][0]);
            Assert.AreEqual(new Feature(4, 2), transposed.x[1][1]);
        
            Assert.AreEqual(new Feature(1, 3), transposed.x[2][0]);
            Assert.AreEqual(new Feature(3, 1), transposed.x[2][1]);
            Assert.AreEqual(new Feature(4, 9), transposed.x[2][2]);
            Assert.AreEqual(new Feature(5, 1), transposed.x[2][3]);
        
            Assert.AreEqual(new Feature(2, 5), transposed.x[3][0]);
            Assert.AreEqual(new Feature(4, 7), transposed.x[3][1]);

            Assert.AreEqual(new Feature(1, 2), transposed.x[4][0]);
            Assert.AreEqual(new Feature(2, 3), transposed.x[4][1]);
            Assert.AreEqual(new Feature(3, 1), transposed.x[4][2]);
            Assert.AreEqual(new Feature(4, 8), transposed.x[4][3]);

            Assert.AreEqual(new Feature(4, 1), transposed.x[5][0]);
        
            Assert.AreEqual(new Feature(2, 4), transposed.x[6][0]);
            Assert.AreEqual(new Feature(4, 5), transposed.x[6][1]);

            Assert.AreEqual(new Feature(2, 2), transposed.x[7][0]);
            Assert.AreEqual(new Feature(4, 4), transposed.x[7][1]);

            Assert.AreEqual(new Feature(3, 7), transposed.x[9][0]);
            Assert.AreEqual(new Feature(5, 3), transposed.x[9][1]);

            Assert.IsTrue(prob.y.SequenceEqual(transposed.y));
        }


        /**
         * compared input/output values with the C version (1.51)
         *
         * IN:
         * res prob.l = 3
         * res prob.n = 4
         * 0: (1,2) (3,1) (4,3)
         * 1: (1,9) (2,7) (3,3) (4,3)
         * 2: (2,1)
         *
         * TRANSPOSED:
         *
         * res prob.l = 3
         *      * res prob.n = 4
         * 0: (1,2) (2,9)
         * 1: (2,7) (3,1)
         * 2: (1,1) (2,3)
         * 3: (1,3) (2,3)
         *
         */
        [TestMethod]
        public void testTranspose3() {


            Problem prob = new Problem();
            prob.l = 3;
            prob.n = 4;
            prob.y = new double[3];
            prob.x = new Feature[4][];
            prob.x[0] = new Feature[3];
            prob.x[1] = new Feature[4];
            prob.x[2] = new Feature[1];
            prob.x[3] = new Feature[1];


            prob.x[0][0] = new Feature(1, 2);
            prob.x[0][1] = new Feature(3, 1);
            prob.x[0][2] = new Feature(4, 3);
            prob.x[1][0] = new Feature(1, 9);
            prob.x[1][1] = new Feature(2, 7);
            prob.x[1][2] = new Feature(3, 3);
            prob.x[1][3] = new Feature(4, 3);


            prob.x[2][0] = new Feature(2, 1);

            prob.x[3][0] = new Feature(3, 2);

            Problem transposed = Linear.transpose(prob);
            Assert.AreEqual(4, transposed.x.Length);
            Assert.AreEqual(2, transposed.x[0].Length);
            Assert.AreEqual(2, transposed.x[1].Length);
            Assert.AreEqual(2, transposed.x[2].Length);
            Assert.AreEqual(2, transposed.x[3].Length);
        
            Assert.AreEqual(new Feature(1, 2) ,transposed.x[0][0]);
            Assert.AreEqual(new Feature(2, 9), transposed.x[0][1]);

            Assert.AreEqual(new Feature(2, 7), transposed.x[1][0]);
            Assert.AreEqual(new Feature(3, 1), transposed.x[1][1]);

            Assert.AreEqual(new Feature(1, 1), transposed.x[2][0]);
            Assert.AreEqual(new Feature(2, 3), transposed.x[2][1]);

            Assert.AreEqual(new Feature(1, 3), transposed.x[3][0]);
            Assert.AreEqual(new Feature(2, 3), transposed.x[3][1]);
        }
    }
}
