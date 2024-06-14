using EDMTranslator.Tokenization;
using EDMTranslator.Translation;
using Xunit.Abstractions;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace EDMTranslatorTest
{
    [Collection("Sequential")]
    public class AihubJaKoTranslatorTest : IDisposable
    {
        private BertJa2GPTTokenizer? tokenizer;
        private AIhubJaKoTranslator? translator;
        private readonly string encoderDictDir;
        private readonly string modelDir;

        private readonly ITestOutputHelper output;

        public AihubJaKoTranslatorTest(ITestOutputHelper output)
        {
            this.output = output;

            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .Build();
            // read test_config.yaml
            using (var reader = new StreamReader("test_config.yaml"))
            {
                var yaml = deserializer.Deserialize<Dictionary<string, AihubJaKoTranslatorConfig>>(reader);
                var config = yaml["aihub_ja_ko_translator"];

                encoderDictDir = config.EncoderDictDir;
                modelDir = config.ModelDir;
            }
        }

        [Fact]
        public async Task TestTranslatorAsync()
        {
            tokenizer ??= await InitTokenizerAsync();
            TestTokenizer(tokenizer);

            translator ??= InitTranslator();
            TestTranslator(translator);
        }

        private async Task<BertJa2GPTTokenizer> InitTokenizerAsync()
        {
            // Prepare the tokenizer
            var encoderVocabPath = await BertJapaneseTokenizer.HuggingFace.GetVocabFromHub("tohoku-nlp/bert-base-japanese-v2");
            var hubName = "skt/kogpt2-base-v2";
            var decoderVocabFilename = "tokenizer.json";
            var decoderVocabPath = await Tokenizers.DotNet.HuggingFace.GetFileFromHub(hubName, decoderVocabFilename, "deps");

            return new BertJa2GPTTokenizer(
                               encoderDictDir: encoderDictDir, encoderVocabPath: encoderVocabPath,
                                              decoderVocabPath: decoderVocabPath);
        }

        private AIhubJaKoTranslator InitTranslator()
        {
            // Prepare the translator
            return new AIhubJaKoTranslator(tokenizer, modelDir);
        }

        private void TestTokenizer(ITokenizer tokenizer)
        {
            output.WriteLine("--Tokenizer test--");
            output.WriteLine("[Encode]");
            var sentenceJa = "打ち合わせが終わった後にご飯を食べましょう。";
            output.WriteLine($"Input: {sentenceJa}");
            var (embeddingsJa, attentionMask) = tokenizer.Encode(sentenceJa);
            output.WriteLine($"Encoded: {string.Join(", ", embeddingsJa)}");

            output.WriteLine("[Decode]");
            // Tokens of "음, 이제 식사도 해볼까요"
            var tokens = new uint[] { 9330, 387, 12857, 9376, 18649, 9098, 7656, 6969, 8084, 1 };
            output.WriteLine($"Input: {string.Join(", ", tokens)}");
            var decoded = tokenizer.Decode(tokens);
            output.WriteLine($"Decoded: {decoded}");
        }

        private void TestTranslator(AIhubJaKoTranslator translator)
        {
            output.WriteLine("--Translator test--");
            Translate(translator, "打ち合わせが終わった後にご飯を食べましょう。");
            Translate(translator, "試験前に緊張したあまり、熱がでてしまった。");
            Translate(translator, "山田は英語にかけてはクラスの誰にも負けない。");
            Translate(translator, "この本によれば、最初の人工橋梁は新石器時代にさかのぼるという。");
        }

        private void Translate(AIhubJaKoTranslator translator, string sentence)
        {
            output.WriteLine($"SourceText: {sentence}");
            string translated = translator.Translate(sentence);
            output.WriteLine($"Translated: {translated}");
        }

        public void Dispose()
        {
            tokenizer = null;
            translator = null;
        }
    }
}
