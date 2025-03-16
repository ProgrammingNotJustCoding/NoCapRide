const SurgeOverlay: React.FC<{
  onOptIn: () => void;
  onCancel: () => void;
}> = ({ onOptIn, onCancel }) => {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/70 animate-fadeIn z-[1000]">
      <div className="bg-gradient-to-br text-yellow-400 from-gray-950 to-gray-900 p-8 rounded-xl shadow-2xl text-center max-w-md mx-4 transform scale-100 animate-scaleIn">
        <img
          src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcGhiZnZnYzM4ZHlkM2ZxcXMwbDI2dTl2NzY5MG8xYXFxeXJxcmp4dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WqCZp6fBYJC91F99ZJ/giphy.gif"
          alt="Surge Hour"
          className="mx-auto mb-4 w-60 h-60"
        />
        <h2 className="text-3xl font-bold text-yellow-400 mb-3">
          Surge Hour Starting!
        </h2>
        <p className="text-white/90 text-lg mb-3">
          Complete 10 surge rides with a{" "}
          <span className="font-bold text-yellow-400">1.3x</span> multiplier on
          all rides! Or go for{" "}
          <span className="font-bold text-yellow-400">2x</span> on normal rides
          with no long-term rewards!
        </p>
        <div className="flex space-x-4 justify-center mt-6">
          <button
            onClick={onOptIn}
            className="bg-yellow-500 hover:bg-yellow-600 text-black font-bold py-2 px-6 rounded-lg transition-colors"
          >
            Opt In
          </button>
          <button
            onClick={onCancel}
            className="bg-gray-700 hover:bg-gray-600 text-white font-bold py-2 px-6 rounded-lg transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default SurgeOverlay;
