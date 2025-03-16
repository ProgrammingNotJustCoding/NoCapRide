"use client";

import Image from "next/image";
import { Switch } from "./ui/switch";
import { useState } from "react";
import Sparkle from "react-sparkle";

type TopDetailsProps = {
  showSparkles: boolean;
};

const TopDetails: React.FC<TopDetailsProps> = ({ showSparkles }) => {
  const [isOnline, setIsOnline] = useState(false);
  return (
    <section className="w-full h-24 flex items-center justify-center py-3 px-2 shadow-md rounded-b-2xl bg-gray-950 text-yellow-400">
      <section className="w-[90%] h-20 flex items-center">
        <div className="flex items-center gap-4 flex-1 ">
          <div className="relative">
            {showSparkles && <Sparkle count={10} fadeOutSpeed={10} />}
            <Image
              src="/harsh.png"
              alt="logo"
              className="w-14 h-14 rounded-full"
              width={32}
              height={32}
            />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-lg font-bold">Harsh Patlu</span>
            <span className="text-md">BajajRE Auto</span>
          </div>
        </div>
        <div className="flex flex-row items-center gap-3">
          {showSparkles && (
            <img
              src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcGhiZnZnYzM4ZHlkM2ZxcXMwbDI2dTl2NzY5MG8xYXFxeXJxcmp4dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/WqCZp6fBYJC91F99ZJ/giphy.gif"
              alt="Surge Hour"
              className="mx-auto mb-4 w-16 h-16"
            />
          )}
          <div className="flex flex-col items-center gap-2 mr-4">
            <Switch
              checked={isOnline}
              onCheckedChange={setIsOnline}
              className="data-[state=checked]:bg-green-500 data-[state=unchecked]:bg-green-200"
            />
            <span className="text-sm">{isOnline ? "Online" : "Offline"}</span>
          </div>
        </div>
      </section>
    </section>
  );
};

export default TopDetails;
