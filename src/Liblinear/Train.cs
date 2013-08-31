// -----------------------------------------------------------------------
// <copyright file="Train.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using System.Diagnostics;
using System.IO;

namespace Liblinear
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
   public class Train
   {
        public static void Main(string[] args)
        {
            new Train().run(args);
        }

        private bool   cross_validation = false;
        private String    inputFilename;
        private String    modelFilename;
        private int       nr_fold;

        public Train()
        {
            Bias = 1;
        }

        private void do_cross_validation() {
            double total_error = 0;
            double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
            double[] target = new double[Problem.l];

            long start, stop;
            start = Environment.TickCount;
            Linear.crossValidation(Problem, Parameter, nr_fold, target);
            stop = Environment.TickCount;
            Console.WriteLine("time: " + (stop - start) + " ms");


            if (Parameter.solverType.isSupportVectorRegression()) {
                for (int i = 0; i < Problem.l; i++) {
                    double y = Problem.y[i];
                    double v = target[i];
                    total_error += (v - y) * (v - y);
                    sumv += v;
                    sumy += y;
                    sumvv += v * v;
                    sumyy += y * y;
                    sumvy += v * y;
                }
                Console.WriteLine("Cross Validation Mean squared error = {0}", total_error / Problem.l);
                Console.WriteLine("Cross Validation Squared correlation coefficient = {0}", //
                    ((Problem.l * sumvy - sumv * sumy) * (Problem.l * sumvy - sumv * sumy)) / ((Problem.l * sumvv - sumv * sumv) * (Problem.l * sumyy - sumy * sumy)));
            } else {
                int total_correct = 0;
                for (int i = 0; i < Problem.l; i++)
                    if (target[i] == Problem.y[i]) ++total_correct;


                Console.WriteLine("correct: {0}", total_correct);
                Console.WriteLine("Cross Validation Accuracy = {0}", 100.0 * total_correct / Problem.l);
            }
        }

        private void exit_with_help() {
            Console.WriteLine("Usage: train [options] training_set_file [model_file]\n" //
                + "options:\n"
                + "-s type : set type of solver (default 1)\n"
                + "  for multi-class classification\n"
                + "    0 -- L2-regularized logistic regression (primal)\n"
                + "    1 -- L2-regularized L2-loss support vector classification (dual)\n"
                + "    2 -- L2-regularized L2-loss support vector classification (primal)\n"
                + "    3 -- L2-regularized L1-loss support vector classification (dual)\n"
                + "    4 -- support vector classification by Crammer and Singer\n"
                + "    5 -- L1-regularized L2-loss support vector classification\n"
                + "    6 -- L1-regularized logistic regression\n"
                + "    7 -- L2-regularized logistic regression (dual)\n"
                + "  for regression\n"
                + "   11 -- L2-regularized L2-loss support vector regression (primal)\n"
                + "   12 -- L2-regularized L2-loss support vector regression (dual)\n"
                + "   13 -- L2-regularized L1-loss support vector regression (dual)\n"
                + "-c cost : set the parameter C (default 1)\n"
                + "-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
                + "-e epsilon : set tolerance of termination criterion\n"
                + "   -s 0 and 2\n" + "       |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
                + "       where f is the primal function and pos/neg are # of\n"
                + "       positive/negative data (default 0.01)\n" + "   -s 11\n"
                + "       |f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
                + "   -s 1, 3, 4 and 7\n" + "       Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
                + "   -s 5 and 6\n"
                + "       |f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
                + "       where f is the primal function (default 0.01)\n"
                + "   -s 12 and 13\n"
                + "       |f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
                + "       where f is the dual function (default 0.1)\n"
                + "-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
                + "-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
                + "-v n: n-fold cross validation mode\n"
                + "-q : quiet mode (no outputs)\n");
            Environment.Exit(1);
        }

        public Problem Problem { get; private set; }

        public double Bias { get; private set; }

        public Parameter Parameter { get; private set; }

        internal void parse_command_line(string[] argv)
        {
            int i;

            // eps: see setting below
            Parameter = new Parameter(SolverType.getById(SolverType.L2R_L2LOSS_SVC_DUAL), 1, Double.PositiveInfinity, 0.1);
            // default values
            Bias = -1;
            cross_validation = false;


            // parse options
            for (i = 0; i < argv.Length; i++) {
                if (argv[i][0] != '-') break;
                if (++i >= argv.Length) exit_with_help();
                switch (argv[i - 1][1]) {
                    case 's':
                        Parameter.solverType = SolverType.getById(Linear.atoi(argv[i]));
                        break;
                    case 'c':
                        Parameter.setC(Linear.atof(argv[i]));
                        break;
                    case 'p':
                        Parameter.setP(Linear.atof(argv[i]));
                        break;
                    case 'e':
                        Parameter.setEps(Linear.atof(argv[i]));
                        break;
                    case 'B':
                        Bias = Linear.atof(argv[i]);
                        break;
                    case 'w':
                        int weightLabel = int.Parse(argv[i - 1].Substring(2));
                        double weight = double.Parse(argv[i]);
                        Parameter.weightLabel = addToArray(Parameter.weightLabel, weightLabel);
                        Parameter.weight = addToArray(Parameter.weight, weight);
                        break;
                    case 'v':
                        cross_validation = true;
                        nr_fold = int.Parse(argv[i]);
                        if (nr_fold < 2) {
                            Console.Error.WriteLine("n-fold cross validation: n must >= 2");
                            exit_with_help();
                        }
                        break;
                    case 'q':
                        i--;
                        Linear.disableDebugOutput();
                        break;
                    default:
                        Console.Error.WriteLine("unknown option");
                        exit_with_help();
                        break;
                }
            }


            // determine filenames


            if (i >= argv.Length) exit_with_help();


            inputFilename = argv[i];


            if (i < argv.Length - 1)
                modelFilename = argv[i + 1];
            else {
                int p = argv[i].LastIndexOf('/');
                ++p; // whew...
                modelFilename = argv[i].Substring(p) + ".model";
            }


            if (Parameter.eps == Double.PositiveInfinity) {
                switch (Parameter.solverType.getId()) {
                    case SolverType.L2R_LR:
                    case SolverType.L2R_L2LOSS_SVC:
                        Parameter.setEps(0.01);
                        break;
                    case SolverType.L2R_L2LOSS_SVR:
                        Parameter.setEps(0.001);
                        break;
                    case SolverType.L2R_L2LOSS_SVC_DUAL:
                    case SolverType.L2R_L1LOSS_SVC_DUAL:
                    case SolverType.MCSVM_CS:
                    case SolverType.L2R_LR_DUAL:
                        Parameter.setEps(0.1);
                        break;
                    case SolverType.L1R_L2LOSS_SVC:
                    case SolverType.L1R_LR:
                        Parameter.setEps(0.01);
                        break;
                    case SolverType.L2R_L1LOSS_SVR_DUAL:
                    case SolverType.L2R_L2LOSS_SVR_DUAL:
                        Parameter.setEps(0.1);
                        break;
                    default:
                        throw new InvalidOperationException("unknown solver type: " + Parameter.solverType);
                }
            }
        }


        /**
         * reads a problem from LibSVM format
         * @param file the SVM file
         * @throws IOException obviously in case of any I/O exception ;)
         * @throws InvalidInputDataException if the input file is not correctly formatted
         */
        public static Problem readProblem(FileInfo file, double bias) {
            using (StreamReader fp = new StreamReader(File.OpenRead(file.FullName)))
            {
                List<Double> vy = new List<Double>();
                List<Feature[]> vx = new List<Feature[]>();
                int max_index = 0;

                int lineNr = 0;

                while (true) {
                    String line = fp.ReadLine();
                    if (line == null) break;
                    lineNr++;

                    var tokens = line.Split(new[]{' ', '\t', '\f', ':'}, StringSplitOptions.RemoveEmptyEntries);
                    if (tokens.Length == 0) {
                        throw new InvalidInputDataException("empty line", file, lineNr);
                    }

                    try {
                        vy.Add(Linear.atof(tokens[0]));
                    } catch (FormatException e) {
                        throw new InvalidInputDataException("invalid label: " + tokens[0], file, lineNr, e);
                    }

                    tokens = tokens.Skip(1).ToArray();
                    int m = tokens.Length / 2;
                    Feature[] x;
                    if (bias >= 0) {
                        x = new Feature[m + 1];
                    } else {
                        x = new Feature[m];
                    }
                    int indexBefore = 0;
                    for (int j = 0; j < m; j++) {
                        var token = tokens[j * 2];
                        int index;
                        try {
                            index = Linear.atoi(token);
                        } catch (FormatException e) {
                            throw new InvalidInputDataException("invalid index: " + token, file, lineNr, e);
                        }

                        // assert that indices are valid and sorted
                        if (index < 0) throw new InvalidInputDataException("invalid index: " + index, file, lineNr);
                        if (index <= indexBefore) throw new InvalidInputDataException("indices must be sorted in ascending order", file, lineNr);
                        indexBefore = index;

                        token = tokens[j * 2 + 1];
                        try {
                            double value = Linear.atof(token);
                            x[j] = new Feature(index, value);
                        } catch (FormatException) {
                            throw new InvalidInputDataException("invalid value: " + token, file, lineNr);
                        }
                    }
                    if (m > 0) {
                        max_index = Math.Max(max_index, x[m - 1].Index);
                    }


                    vx.Add(x);
                }


                return constructProblem(vy, vx, max_index, bias);
            }
        }


        public void readProblem(string filename)  {
            Problem = Train.readProblem(new FileInfo(filename), Bias);
        }


        private static int[] addToArray(int[] array, int newElement) {
            int length = array != null ? array.Length : 0;
            int[] newArray = new int[length + 1];
            if (array != null && length > 0) {
               Array.Copy(array, 0, newArray, 0, length);
            }
            newArray[length] = newElement;
            return newArray;
        }


        private static double[] addToArray(double[] array, double newElement) {
            int length = array != null ? array.Length : 0;
            double[] newArray = new double[length + 1];
            if (array != null && length > 0) {
                Array.Copy(array, 0, newArray, 0, length);
            }
            newArray[length] = newElement;
            return newArray;
        }


        private static Problem constructProblem(List<Double> vy, List<Feature[]> vx, int max_index, double bias) {
            Problem prob = new Problem();
            prob.bias = bias;
            prob.l = vy.Count;
            prob.n = max_index;
            if (bias >= 0) {
                prob.n++;
            }
            prob.x = new Feature[prob.l][];
            for (int i = 0; i < prob.l; i++) {
                prob.x[i] = vx[i];


                if (bias >= 0) {
                    Debug.Assert(prob.x[i][prob.x[i].Length - 1] == null);
                    prob.x[i][prob.x[i].Length - 1] = new Feature(max_index + 1, bias);
                }
            }


            prob.y = new double[prob.l];
            for (int i = 0; i < prob.l; i++)
                prob.y[i] = vy[i];


            return prob;
        }


        private void run(String[] args) {
            parse_command_line(args);
            readProblem(inputFilename);
            if (cross_validation)
                do_cross_validation();
            else {
                Model model = Linear.train(Problem, Parameter);
                Linear.saveModel(new FileInfo(modelFilename), model);
            }
        }
    }
}
