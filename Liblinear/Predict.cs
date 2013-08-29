// -----------------------------------------------------------------------
// <copyright file="Predict.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Liblinear
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Diagnostics;
    using System.IO;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public class Predict
    {
        private static bool flag_predict_probability = false;

        /**
         * <p><b>Note: The streams are NOT closed</b></p>
         */
        public static void doPredict(StreamReader reader, StreamWriter writer, Model model) {
        int correct = 0;
        int total = 0;
        double error = 0;
        double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

        int nr_class = model.getNrClass();
        double[] prob_estimates = null;
        int n;
        int nr_feature = model.getNrFeature();
        if (model.bias >= 0)
            n = nr_feature + 1;
        else
            n = nr_feature;

        if (flag_predict_probability && !model.isProbabilityModel()) {
            throw new ArgumentException("probability output is only supported for logistic regression");
        }

        if (flag_predict_probability) {
            int[] labels = model.getLabels();
            prob_estimates = new double[nr_class];

            writer.Write("labels");
            for (int j = 0; j < nr_class; j++)
                writer.Write(" {0}", labels[j]);
            writer.WriteLine();
        }

        String line = null;
        while ((line = reader.ReadLine()) != null) {
            List<Feature> x = new List<Feature>();
            string[] parts = line.Split(new[]{' ', '\t'}, StringSplitOptions.RemoveEmptyEntries);
            double target_label;
            if (parts.Length == 0)
            {
                throw new InvalidOperationException("Wrong input format at line " + (total + 1));
            }

            String label = parts[0];
            target_label = Linear.atof(label);

            foreach (var token in parts.Skip(1))
            {
                string[] split = token.Split(':');
                if (split.Length < 2) {
                    throw new InvalidOperationException("Wrong input format at line " + (total + 1));
                }

                try {
                    int idx = Linear.atoi(split[0]);
                    double val = Linear.atof(split[1]);

                    // feature indices larger than those in training are not used
                    if (idx <= nr_feature) {
                        Feature node = new Feature(idx, val);
                        x.Add(node);
                    }
                } catch (FormatException e) {
                    throw new InvalidOperationException("Wrong input format at line " + (total + 1), e);
                }
            }

            if (model.bias >= 0) {
                Feature node = new Feature(n, model.bias);
                x.Add(node);
            }

            Feature[] nodes = x.ToArray();

            double predict_label;

            if (flag_predict_probability) {
                Debug.Assert(prob_estimates != null);
                predict_label = Linear.predictProbability(model, nodes, prob_estimates);
                Console.Write(predict_label);
                for (int j = 0; j < model.nr_class; j++)
                    Console.Write(" {0}", prob_estimates[j]);
                Console.WriteLine();
            } else {
                predict_label = Linear.predict(model, nodes);
                Console.WriteLine("{0}", predict_label);
            }

            if (predict_label == target_label) {
                ++correct;
            }

            error += (predict_label - target_label) * (predict_label - target_label);
            sump += predict_label;
            sumt += target_label;
            sumpp += predict_label * predict_label;
            sumtt += target_label * target_label;
            sumpt += predict_label * target_label;
            ++total;
        }

        if (model.solverType.isSupportVectorRegression()) //
        {
            Linear.info("Mean squared error = {0} (regression)", error / total);
            Linear.info("Squared correlation coefficient = {0} (regression)", //
                ((total * sumpt - sump * sumt) * (total * sumpt - sump * sumt)) / ((total * sumpp - sump * sump) * (total * sumtt - sumt * sumt)));
        } else {
            Linear.info("Accuracy = {0} ({1}/{2})", (double)correct / total * 100, correct, total);
        }
    }

        private static void exit_with_help()
        {
            Console.WriteLine("Usage: predict [options] test_file model_file output_file");
            Console.WriteLine("options:");
            Console.WriteLine(
                "-b probability_estimates: whether to output probability estimates, 0 or 1 (default 0); currently for logistic regression only");
            Console.WriteLine("-q quiet mode (no outputs)");
            Environment.Exit(1);
        }

        public static void main(String[] argv)
        {
            int i;

            // parse options
            for (i = 0; i < argv.Length; i++)
            {
                if (argv[i][0] != '-') break;
                ++i;
                switch (argv[i - 1][1])
                {
                    case 'b':
                        try
                        {
                            flag_predict_probability = (Linear.atoi(argv[i]) != 0);
                        }
                        catch (FormatException e)
                        {
                            exit_with_help();
                        }
                        break;

                    case 'q':
                        i--;
                        Linear.disableDebugOutput();
                        break;

                    default:
                        Console.Error.WriteLine("unknown option: -{0}", argv[i - 1][1]);
                        exit_with_help();
                        break;
                }
            }
            if (i >= argv.Length || argv.Length <= i + 2)
            {
                exit_with_help();
            }


            using (var reader = new StreamReader(File.OpenRead(argv[i]), Linear.FILE_CHARSET))
            using (var writer = new StreamWriter(File.OpenWrite(argv[i + 2]), Linear.FILE_CHARSET))
            {
                Model model = Linear.loadModel(new FileInfo(argv[i + 1]));
                doPredict(reader, writer, model);
            }
        }
    }
}
