// -----------------------------------------------------------------------
// <copyright file="IrisDataset.cs" company="Sharpkit.Learn">
// # Copyright (c) 2007 David Cournapeau &lt;cournape@gmail.com>
//   2010 Fabian Pedregosa &lt;fabian.pedregosa@inria.fr>
//   2010 Olivier Grisel &lt;olivier.grisel@ensta.org>
//   2013 Sergey Zyuzin &lt;forever.zet@gmail.com>
// License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------
using System.IO;
using System.Linq;
using System.Reflection;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Sharpkit.Learn.Datasets
{
    using System;

    /// <summary>
    /// The iris dataset is a classic and very easy multi-class classification dataset.
    /// </summary>
    public class IrisDataset
    {
        /// <summary>
        /// Gets the data to learn.
        /// </summary>
        public Matrix Data { get; private set; }

        /// <summary>
        /// Gets the classification labels.
        /// </summary>
        public int[] Target { get; private set; }

        /// <summary>
        /// Gets the meaning of the labels.
        /// </summary>
        public string[] TargetNames { get; private set; }

        /// <summary>
        /// Gets the meaning of the features.
        /// </summary>
        public string[] FeatureNames { get; private set; }

        /// <summary>
        /// Gets or sets description of the dataset.
        /// </summary>
        public string Descr { get; set; }

        /// <summary>
        /// Load and return the iris dataset (classification).
        /// =================   ==============
        /// Classes                          3
        /// Samples per class               50
        /// Samples total                  150
        /// Dimensionality                   4
        /// Features            real, positive
        /// =================   ==============
        /// </summary>
        /// <returns>Instance of <see cref="IrisDataset"/> class.</returns>
        public static IrisDataset Load()
        {
            var assembly = Assembly.GetAssembly(typeof(IrisDataset));
            string descr;
            using (var sr = new StreamReader(assembly.GetManifestResourceStream("Sharpkit.Learn.Datasets.Data.iris.csv")))
            {
                descr = sr.ReadToEnd();
            }

            using (var datastream = assembly.GetManifestResourceStream("Sharpkit.Learn.Datasets.Data.iris.csv"))
            {
                var sr = new StreamReader(datastream);
                var l = sr.ReadLine();
                var parts = l.Split(',');
                int numSamples = int.Parse(parts[0]);
                int numFeatures = int.Parse(parts[1]);
                string[] targetNames = parts.Skip(2).ToArray();
                string line;

                DenseMatrix data = new DenseMatrix(numSamples, numFeatures);
                int[] target = new int[numSamples];
                int i = 0;
                while ((line = sr.ReadLine()) != null)
                {
                    parts = line.Split(',');
                    for (int j = 0; j < parts.Length - 1; j++)
                    {
                        data[i, j] = double.Parse(parts[j]);
                    }

                    target[i] = int.Parse(parts.Last());

                    i++;
                }

                return new IrisDataset
                           {
                               Data = data,
                               Target = target,
                               TargetNames = targetNames,
                               Descr = descr,
                               FeatureNames = new[]
                                                  {
                                                      "sepal length (cm)",
                                                      "sepal width (cm)",
                                                      "petal length (cm)",
                                                      "petal width (cm)"
                                                  }
                           };
            }
        }
    }
}
