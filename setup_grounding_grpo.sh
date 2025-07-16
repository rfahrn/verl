#!/bin/bash
set -e

echo "🚀 Setting up VeRL GRPO Training for Grounding Tasks"

# Step 1: Install VeRL
echo "📦 Installing VeRL..."
if [ ! -d "verl" ]; then
    git clone https://github.com/volcengine/verl.git
fi
cd verl
pip install -r requirements.txt
pip install -e .

# Optional: Install additional backends
echo "🔧 Installing additional backends..."
pip install vllm  # for fast inference
pip install sglang  # alternative inference backend

# Step 2: Create data directory
echo "📁 Creating data directory..."
mkdir -p ~/data/grounding

# Step 3: Copy configuration files
echo "📄 Setting up configuration files..."
cp ../run_grounding_grpo.sh .
cp ../custom_grounding_reward.py .
chmod +x run_grounding_grpo.sh

# Step 4: Create environment setup
echo "🔧 Creating environment setup..."
cat > setup_env.sh << 'EOF'
#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=grounding_grpo_training
export WANDB_ENTITY=your_username  # Change this to your W&B username
EOF

chmod +x setup_env.sh

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run your data preparation script:"
echo "   python radvlm/data/create_grounding_verl_corrected.py"
echo ""
echo "2. Update paths in run_grounding_grpo.sh to point to your data"
echo ""
echo "3. Setup environment and run training:"
echo "   source setup_env.sh"
echo "   ./run_grounding_grpo.sh"
echo ""
echo "📊 Monitor training progress with:"
echo "   tail -f logs/grounding_grpo_training.log"
echo "   # or check your W&B dashboard"