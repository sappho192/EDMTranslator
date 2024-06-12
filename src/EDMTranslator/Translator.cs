namespace EDMTranslator
{
    public abstract class Translator : ITranslator
    {
        public abstract string Translate(string sentence);
        public ModelInfo ModelInfo { get; }

        protected Translator(ModelInfo modelInfo)
        {
            ModelInfo = modelInfo;
        }
    }
}
