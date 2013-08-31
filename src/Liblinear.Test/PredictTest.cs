// -----------------------------------------------------------------------
// <copyright file="PredictTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Liblinear.Test
{
    using System;
    using System.Text;
    using System.IO;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    [TestClass]
    public class PredictTest
    {
        private Model testModel = LinearTest.createRandomModel();
        private StringBuilder sb = new StringBuilder();
        private StreamWriter writer = new StreamWriter(new MemoryStream());


        [TestInitialize]
        public void setUp()
        {
            //System.setOut(mock(PrintStream.class)); // dev/null
            Assert.IsTrue(testModel.getNrClass() >= 2);
            Assert.IsTrue(testModel.getNrFeature() >= 10);
        }


        private void testWithLines(StringBuilder sb)
        {
            var reader = new StreamReader(new MemoryStream(Encoding.UTF8.GetBytes(sb.ToString())));


            Predict.doPredict(reader, writer, testModel);
        }


        [TestMethod]
        [ExpectedException(typeof (InvalidOperationException))]
        public void testDoPredictCorruptLine()
        {
            sb.Append(testModel.label[0]).Append(" abc").Append("\n");
            testWithLines(sb);
        }


        [TestMethod]
        [ExpectedException(typeof (ArgumentException))]
        public void testDoPredictCorruptLine2()
        {
            sb.Append(testModel.label[0]).Append(" 1:").Append("\n");
            testWithLines(sb);
        }


        [TestMethod]
        public void testDoPredict()
        {
            sb.Append(testModel.label[0]).Append(" 1:0.32393").Append("\n");
            sb.Append(testModel.label[1]).Append(" 2:-71.555   9:88223").Append("\n");
            testWithLines(sb);
        }
    }
}
