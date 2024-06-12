using EDMTranslator;

var encoderVocabPath = await BertJapaneseTokenizer.HuggingFace.GetVocabFromHub("tohoku-nlp/bert-base-japanese-v2");
// Prepare the tokenizer for decoding
var hubName = "skt/kogpt2-base-v2";
var decoderVocabFilename = "tokenizer.json";
var decoderVocabPath = await Tokenizers.DotNet.HuggingFace.GetFileFromHub(hubName, decoderVocabFilename, "deps");

var tokenizer = new BertJa2KoGPTTokenizer(
    encoderDictDir: @"D:\DATASET\unidic-mecab-2.1.2_bin", encoderVocabPath: encoderVocabPath,
    decoderVocabPath: decoderVocabPath);

Console.WriteLine("--Tokenizer test--");
Console.WriteLine("[Encode]");
var sentenceJa = "打ち合わせが終わった後にご飯を食べましょう。";
Console.WriteLine($"Input: {sentenceJa}");
(var embeddingsJa, var attentionMask) = tokenizer.Encode(sentenceJa);
Console.WriteLine($"Encoded: {string.Join(", ", embeddingsJa)}");

Console.WriteLine("[Decode]");
var tokens = new uint[] { 9330, 387, 12857, 9376, 18649, 9098, 7656, 6969, 8084, 1 };
Console.WriteLine($"Input: {string.Join(", ", tokens)}");
var decoded = tokenizer.Decode(tokens);
Console.WriteLine($"Decoded: {decoded}");