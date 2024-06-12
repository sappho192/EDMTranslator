namespace EDMTranslator
{
    public class ModelInfo
    {
        public string ModelName { get; }
        public string ModelHubName { get; }
        public ModelType ModelType { get; }
        public string ModelUri { get; }

        public ModelInfo(string modelName, string modelHubName, ModelType modelType, string modelUri)
        {
            ModelName = modelName;
            ModelHubName = modelHubName;
            ModelType = modelType;
            ModelUri = modelUri;
        }
    }
}
