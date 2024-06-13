using EDMTranslator.Models;

namespace EDMTranslator
{
    public sealed class EncoderDecoderModelInfo : ModelInfo
    {
        public ModelInfo EncoderModelInfo { get; }
        public ModelInfo DecoderModelInfo { get; }

        public EncoderDecoderModelInfo(ModelInfo encoderModelInfo, ModelInfo decoderModelInfo, 
            string modelHubName, string modelUri)
            : base(modelName: $"{encoderModelInfo.ModelName}-{decoderModelInfo.ModelName}",
                  modelHubName: modelHubName, modelType: ModelType.EncoderDecoder, 
                  modelUri: modelUri)
        {
            EncoderModelInfo = encoderModelInfo;
            DecoderModelInfo = decoderModelInfo;
        }
    }
}
