// -----------------------------------------------------------------------
// <copyright file="InvalidDataException.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

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
   public class InvalidInputDataException : Exception {


    // private static long serialVersionUID = 2945131732407207308L;


    private int         _line;


    private FileInfo              _file;


    public InvalidInputDataException( String message, FileInfo file, int line ) :base(message) {
        _file = file;
        _line = line;
    }


    public InvalidInputDataException(String message, String filename, int line)
        : this(message, new FileInfo(filename), line)
    {
    }


    public InvalidInputDataException( String message, FileInfo file, int lineNr, Exception cause ) : base(message, cause){
        _file = file;
        _line = lineNr;
    }


    public InvalidInputDataException( String message, String filename, int lineNr, Exception cause ) : this(message, new FileInfo(filename), lineNr, cause) {
        
    }


    public FileInfo getFile() {
        return _file;
    }


    /**
     * This methods returns the path of the file.
     * The method name might be misleading.
     *
     * @deprecated use {@link #getFile()} instead
     */
    public String getFilename() {
        return _file.FullName;
    }


    public int getLine() {
        return _line;
    }

    public override string ToString() {
        return base.ToString() + " (" + _file + ":" + _line + ")";
    }
   }
}
