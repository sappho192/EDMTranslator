using EDMTranslator.Tokenization;
using EDMTranslator.Translation;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace EDMTranslatorTest
{
    public class FF14JaKoTranslatorTest
    {
        private BertJa2GPTTokenizer? tokenizer;
        private FF14JaKoTranslator? translator;
        private readonly string encoderDictDir;
        private readonly string modelDir;

        public FF14JaKoTranslatorTest()
        {
            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .Build();
            // read test_config.yaml
            using (var reader = new StreamReader("test_config.yaml"))
            {
                var yaml = deserializer.Deserialize<Dictionary<string, FF14JaKoTranslatorConfig>>(reader);
                var config = yaml["ff14_ja_ko_translator"];

                encoderDictDir = config.EncoderDictDir;
                modelDir = config.ModelDir;
            }
        }

        [Fact]
        public async Task TestTokenizerAsync()
        {
            tokenizer ??= await InitTokenizerAsync();
            TestTokenizer(tokenizer);
        }

        [Fact]
        public async Task TestTranslatorAsync()
        {
            tokenizer ??= await InitTokenizerAsync();
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

        private FF14JaKoTranslator InitTranslator()
        {
            // Prepare the translator
            return new FF14JaKoTranslator(tokenizer, modelDir);
        }

        private static void TestTokenizer(ITokenizer tokenizer)
        {
            Console.WriteLine("--Tokenizer test--");
            Console.WriteLine("[Encode]");
            var sentenceJa = "打ち合わせが終わった後にご飯を食べましょう。";
            Console.WriteLine($"Input: {sentenceJa}");
            var (embeddingsJa, attentionMask) = tokenizer.Encode(sentenceJa);
            Console.WriteLine($"Encoded: {string.Join(", ", embeddingsJa)}");

            Console.WriteLine("[Decode]");
            // Tokens of "음, 이제 식사도 해볼까요"
            var tokens = new uint[] { 9330, 387, 12857, 9376, 18649, 9098, 7656, 6969, 8084, 1 };
            Console.WriteLine($"Input: {string.Join(", ", tokens)}");
            var decoded = tokenizer.Decode(tokens);
            Console.WriteLine($"Decoded: {decoded}");
        }

        private static void TestTranslator(FF14JaKoTranslator translator)
        {
            Console.WriteLine("--Translator test--");
            Translate(translator, "打ち合わせが終わった後にご飯を食べましょう。");
            Translate(translator, "試験前に緊張したあまり、熱がでてしまった。");
            Translate(translator, "山田は英語にかけてはクラスの誰にも負けない。");
            Translate(translator, "この本によれば、最初の人工橋梁は新石器時代にさかのぼるという。");
        }

        private static void Translate(FF14JaKoTranslator translator, string sentence)
        {
            Console.WriteLine($"SourceText: {sentence}");
            string translated = translator.Translate(sentence);
            Console.WriteLine($"Translated: {translated}");
        }
    }
}