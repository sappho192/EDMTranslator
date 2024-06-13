using EDMTranslator.Models;
using EDMTranslator.Tokenization;

namespace EDMTranslator.Translation
{
    public sealed class FF14Translator : EncoderDecoderTranslator
    {
        public FF14Translator(BertJa2KoGPTTokenizer tokenizer) : base(
            modelName: "ffxiv-ja-ko-translator",
            modelHubName: "sappho192/ffxiv-ja-ko-translator",
            tokenizer: tokenizer)
        {

        }

        public override string Translate(string sentence)
        {
            return sentence;
        }
    }
}
