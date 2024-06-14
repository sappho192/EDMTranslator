using EDMTranslator.Models;
using EDMTranslator.Tokenization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;

namespace EDMTranslator.Translation
{
    public class JESCJaEnTranslator : EncoderDecoderTranslator
    {
        private readonly InferenceSession encoderSession;
        private readonly InferenceSession decoderSession;

        public JESCJaEnTranslator(BertJa2GPTTokenizer tokenizer, string modelDirPath) : base(
            modelName: "jesc-ja-en-translator",
            modelHubName: "sappho192/jesc-ja-en-translator",
            tokenizer: tokenizer)
        {
            string encoderModelPath = Path.Combine(modelDirPath, "encoder_model.onnx");
            string decoderModelPath = Path.Combine(modelDirPath, "decoder_model_merged.onnx");
            var sessionOptions = new SessionOptions
            {
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
            };
            encoderSession = new(encoderModelPath, sessionOptions);
            decoderSession = new(decoderModelPath, sessionOptions);
        }

        public override string Translate(string sentence)
        {
            (var inputIds, var attentionMask) = Tokenizer.Encode(sentence);
            //Console.WriteLine($"Input tokens: {string.Join(", ", inputIds)}");

            // inputIds to NDArray
            NDArray ndInputIds = np.array(inputIds);
            ndInputIds = np.expand_dims(ndInputIds, 0);
            NDArray ndAttentionMask = np.array(attentionMask.ToArray());
            ndAttentionMask = np.expand_dims(ndAttentionMask, 0);

            var encoderInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", ndInputIds.ToMuliDimArray<long>().ToTensor<long>()),
                NamedOnnxValue.CreateFromTensor("attention_mask", ndAttentionMask.ToMuliDimArray<long>().ToTensor<long>())
            };

            var encoderResults = encoderSession.Run(encoderInput);
            var encoderResult = encoderResults[0];

            var singleBoolArray = new bool[] { false };
            var useCacheBranch = np.array(singleBoolArray);
            //useCacheBranch = np.expand_dims(useCacheBranch, 0);

            //var zeros = np.zeros<float>(1, 12, inputIds.Length, 64);
            var decoderInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", ndInputIds.ToMuliDimArray<long>().ToTensor<long>()),
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderResult.AsTensor<float>()),
                NamedOnnxValue.CreateFromTensor("use_cache_branch", useCacheBranch.ToMuliDimArray<bool>().ToTensor<bool>())
            };

            var generatedText = GreedySearch(decoderInput, decoderSession);
            return generatedText;
        }
    }
}
