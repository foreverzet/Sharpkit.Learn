// -----------------------------------------------------------------------
// <copyright file="IrisDataset.cs" company="Sharpkit.Learn">
// # Copyright (c) 2007 David Cournapeau &lt;cournape@gmail.com>
//   2010 Fabian Pedregosa &lt;fabian.pedregosa@inria.fr>
//   2010 Olivier Grisel &lt;olivier.grisel@ensta.org>
//   2013 Sergey Zyuzin &lt;forever.zet@gmail.com>
// License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

using MathNet.Numerics.LinearAlgebra.Double;

namespace Sharpkit.Learn.Datasets
{
    using System;
    using System.IO;
    using System.Linq;
    using System.Reflection;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// The iris dataset is a classic and very easy multi-class classification dataset.
    /// </summary>
    public class BostonDataset
    {
        /// <summary>
        /// Gets the data to learn.
        /// </summary>
        public Matrix<double> Data { get; private set; }

        /// <summary>
        /// Gets the classification labels.
        /// </summary>
        public Vector<double> Target { get; private set; }

        /// <summary>
        /// Gets the meaning of the features.
        /// </summary>
        public string[] FeatureNames { get; private set; }

        /// <summary>
        /// Gets or sets description of the dataset.
        /// </summary>
        public string Descr { get; set; }

        /// <summary>
        /// Load and return the boston house-prices dataset (regression).
        ///
        /// ==============     ==============
        /// Samples total                 506
        /// Dimensionality                 13
        /// Features           real, positive
        /// Targets             real 5. - 50.
        /// ==============     ==============
        /// </summary>
        /// <returns></returns>
        public static BostonDataset Load()
        {

            var assembly = Assembly.GetAssembly(typeof(IrisDataset));
            string descr;
            using (var sr = new StreamReader(assembly.GetManifestResourceStream("Sharpkit.Learn.Datasets.Data.boston_house_prices.rst")))
            {
                descr = sr.ReadToEnd();
            }

            using (var datastream = assembly.GetManifestResourceStream("Sharpkit.Learn.Datasets.Data.boston_house_prices.csv"))
            {
                var sr = new StreamReader(datastream);
                var line = sr.ReadLine();
                var parts = line.Split(',');
                int nSamples = int.Parse(parts[0]);
                int nFeatures = int.Parse(parts[1]);

                line = sr.ReadLine();
                var featureNames = line.Split(',');

                DenseMatrix data = new DenseMatrix(nSamples, nFeatures);
                Vector<double> target = new DenseVector(nSamples);
                int i = 0;
                while ((line = sr.ReadLine()) != null)
                {
                    parts = line.Split(',');
                    for (int j = 0; j < parts.Length - 1; j++)
                    {
                        data[i, j] = double.Parse(parts[j]);
                    }

                    target[i] = double.Parse(parts.Last());

                    i++;
                }

                return new BostonDataset
                           {
                               Data = data,
                               Target = target,
                               //TargetNames = targetNames,
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
