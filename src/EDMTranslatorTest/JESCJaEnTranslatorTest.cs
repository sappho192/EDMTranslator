using EDMTranslator.Tokenization;
using EDMTranslator.Translation;
using Xunit.Abstractions;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace EDMTranslatorTest
{
    [Collection("Sequential")]
    public class JESCJaEnTranslatorTest
    {
        private BertJa2GPTTokenizer? tokenizer;
        private JESCJaEnTranslator? translator;
        private readonly string encoderDictDir;
        private readonly string modelDir;

        private readonly ITestOutputHelper output;

        public JESCJaEnTranslatorTest(ITestOutputHelper output)
        {
            this.output = output;

            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .Build();
            // read test_config.yaml
            using (var reader = new StreamReader("test_config.yaml"))
            {
                var yaml = deserializer.Deserialize<Dictionary<string, JESCJaEnTranslatorConfig>>(reader);
                var config = yaml["jesc_ja_en_translator"];

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
            var hubName = "openai-community/gpt2";
            var decoderVocabFilename = "tokenizer.json";
            var decoderVocabPath = await Tokenizers.DotNet.HuggingFace.GetFileFromHub(hubName, decoderVocabFilename, "deps");

            return new BertJa2GPTTokenizer(
                               encoderDictDir: encoderDictDir, encoderVocabPath: encoderVocabPath,
                                              decoderVocabPath: decoderVocabPath);
        }

        private JESCJaEnTranslator InitTranslator()
        {
            return new JESCJaEnTranslator(tokenizer, modelDir);
        }

        private void TestTokenizer(BertJa2GPTTokenizer tokenizer)
        {
            output.WriteLine("--Tokenizer test--");
            output.WriteLine("[Encode]");
            var sentenceJa = "打ち合わせが終わった後にご飯を食べましょう。";
            output.WriteLine($"Input: {sentenceJa}");
            var (embeddingsJa, attentionMask) = tokenizer.Encode(sentenceJa);
            output.WriteLine($"Encoded: {string.Join(", ", embeddingsJa)}");

            output.WriteLine("[Decode]");
            // Tokens of "i was nervous before the exam, and i had a fever."
            var tokens = new uint[] { 72, 373, 10927, 878, 262, 2814, 11, 290, 1312, 550, 257, 17372, 13 };
            output.WriteLine($"Input: {string.Join(", ", tokens)}");
            var decoded = tokenizer.Decode(tokens);
            output.WriteLine($"Decoded: {decoded}");
        }

        private void TestTranslator(JESCJaEnTranslator translator)
        {
            output.WriteLine("--Translator test--");
            Translate(translator, "打ち合わせが終わった後にご飯を食べましょう。");
            Translate(translator, "試験前に緊張したあまり、熱がでてしまった。");
            Translate(translator, "山田は英語にかけてはクラスの誰にも負けない。");
            Translate(translator, "この本によれば、最初の人工橋梁は新石器時代にさかのぼるという。");
        }

        private void Translate(JESCJaEnTranslator translator, string sentence)
        {
            output.WriteLine($"SourceText: {sentence}");
            string translated = translator.Translate(sentence);
            output.WriteLine($"Translated: {translated}");
        }
    }
}
