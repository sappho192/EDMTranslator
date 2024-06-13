using EDMTranslator.Models;
using EDMTranslator.Tokenization;

namespace EDMTranslator.Translation
{
    public abstract class Translator : ITranslator
    {
        public abstract string Translate(string sentence);
        public ModelInfo ModelInfo { get; }
        public ITokenizer Tokenizer { get; }

        protected Translator(ModelInfo modelInfo, ITokenizer tokenizer)
        {
            ModelInfo = modelInfo;
            Tokenizer = tokenizer;
        }
    }
}
