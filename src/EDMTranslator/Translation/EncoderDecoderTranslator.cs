using EDMTranslator.Tokenization;
using EDMTranslator.Translation;

namespace EDMTranslator.Models
{
    public abstract class EncoderDecoderTranslator : Translator
    {
        public EncoderDecoderTranslator(string modelName, string modelHubName, ITokenizer tokenizer) :
            base(modelInfo: new ModelInfo(
                modelName: modelName,
                modelHubName: modelHubName,
                modelType: ModelType.EncoderDecoder,
                modelUri: $"https://huggingface.co/{modelHubName}"),
            tokenizer)
        {

        }
    }
}
