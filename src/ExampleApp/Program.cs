using EDMTranslator.Tokenization;
using EDMTranslator.Translation;

// Prepare the tokenizer
var encoderVocabPath = await BertJapaneseTokenizer.HuggingFace.GetVocabFromHub("tohoku-nlp/bert-base-japanese-v2");
var hubName = "skt/kogpt2-base-v2";
var decoderVocabFilename = "tokenizer.json";
var decoderVocabPath = await Tokenizers.DotNet.HuggingFace.GetFileFromHub(hubName, decoderVocabFilename, "deps");

string encoderDictDir = @"D:\DATASET\unidic-mecab-2.1.2_bin";
var tokenizer = new BertJa2KoGPTTokenizer(
    encoderDictDir: encoderDictDir, encoderVocabPath: encoderVocabPath,
    decoderVocabPath: decoderVocabPath);

void TestTokenizer(ITokenizer tokenizer)
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
TestTokenizer(tokenizer);

// Prepare the translator
string modelDir = @"D:\MODEL\ffxiv-ja-ko-translator\onnx";
var translator = new FF14Translator(tokenizer, modelDir);
void TestTranslator(FF14Translator translator)
{
    Console.WriteLine("--Translator test--");
    Translate(translator, "打ち合わせが終わった後にご飯を食べましょう。");
    Translate(translator, "試験前に緊張したあまり、熱がでてしまった。");
    Translate(translator, "山田は英語にかけてはクラスの誰にも負けない。");
    Translate(translator, "この本によれば、最初の人工橋梁は新石器時代にさかのぼるという。");
}
TestTranslator(translator);

static void Translate(FF14Translator translator, string sentence)
{
    Console.WriteLine($"SourceText: {sentence}");
    string translated = translator.Translate(sentence);
    Console.WriteLine($"Translated: {translated}");
}